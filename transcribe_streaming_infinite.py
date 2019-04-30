#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:

    pip install pyaudio
    pip install termcolor

Example usage:
    python transcribe_streaming_infinite.py
"""

# [START speech_transcribe_infinite_streaming]

import time
import re
import sys
import os

from google.cloud import speech_v1p1beta1 as speech
import pyaudio
from six.moves import queue

#get num of columns in terminal to truncate transcriptions that are too long
rows, columns = os.popen('stty size', 'r').read().split()
print(columns)

# Audio recording parameters
STREAMING_LIMIT = 10000
WORKAROUND_WINDOW = 200
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED   = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'

def get_current_time():
    return int(round(time.time() * 1000))

class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk_size):
        self._rate = rate
        self._chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()

        self._closed = True
        self._start_time = get_current_time()
        self._restart_counter = 0
        self._audio_input = []
        self._last_audio_input = []
        self._result_end_time = 0
        self._is_final_end_time = 0
        self._final_request_end_time = 0
        self._bridging_offset = 0
        self._last_transcript_was_final = False
        self._new_stream = True

    def __enter__(self):
        self._closed = False

        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):

        while not self._closed:
            data = []

            if get_current_time() - self._start_time > STREAMING_LIMIT:
                self._start_time = get_current_time()
                break

            if self._new_stream and (len(self._last_audio_input) != 0):

                chunkTime = STREAMING_LIMIT / len(self._last_audio_input)

                if chunkTime != 0:

                    if self._bridging_offset < 0:
                        self._bridging_offset = 0

                    if self._bridging_offset > self._final_request_end_time:
                        self._bridging_offset = self._final_request_end_time

                    chunksFromMS = round((self._final_request_end_time - self._bridging_offset) / chunkTime)

                    self._bridging_offset = round((len(self._last_audio_input) - chunksFromMS) * chunkTime)

                    for i in range(chunksFromMS, len(self._last_audio_input)):
                        data.append(self._last_audio_input[i])

                self._new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self._audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self._audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b''.join(data)



def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """

    for response in responses:

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_nanos = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.nanos:
            result_nanos = result.result_end_time.nanos

        stream._result_end_time = int((result_seconds * 1000) + (result_nanos / 1000000))

        corrected_time = stream._result_end_time - stream._bridging_offset + (STREAMING_LIMIT * stream._restart_counter);
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:

            if stream._result_end_time < (STREAMING_LIMIT - WORKAROUND_WINDOW):
                sys.stdout.write(GREEN)
                sys.stdout.write('\033[K')
                sys.stdout.write(str(corrected_time) + ': ' + transcript + '\n')

                stream._is_final_end_time = stream._result_end_time
                stream._last_transcript_was_final = True

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if re.search(r'\b(exit|quit)\b', transcript, re.I):
                    sys.stdout.write(YELLOW)
                    sys.stdout.write('Exiting...\n')
                    stream._closed = True
                    break

        else:
            sys.stdout.write(RED)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\r')

            stream._last_transcript_was_final = False

def main():

    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US',
        max_alternatives=1)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager._chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n');
    sys.stdout.write('=========================================================\n');

    with mic_manager as stream:

        while not stream._closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write('\n' + str(STREAMING_LIMIT * stream._restart_counter) + ': NEW REQUEST\n')

            stream._audio_input = []
            audio_generator = stream.generator()
            temporary_generator = stream.generator()

            requests = (speech.types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)



            responses = client.streaming_recognize(streaming_config,
                                                   requests)


            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream)

            if stream._result_end_time > 0:
                stream._final_request_end_time = stream._is_final_end_time
            stream._result_end_time = 0
            stream._last_audio_input = []
            stream._last_audio_input = stream._audio_input
            stream._audio_input = []
            stream._restart_counter = stream._restart_counter + 1

            if not stream._last_transcript_was_final:
                sys.stdout.write('\n');
            stream._new_stream = True

if __name__ == '__main__':
    main()
# [END speech_transcribe_infinite_streaming]
