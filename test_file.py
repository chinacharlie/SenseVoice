import os
import sys
import requests
import threading
import time
import librosa
import numpy as np


def run(i: int):
    j = 0
    while (j < 1):
        t1 = time.time()
        filename = '4.mp3'

        file = open(filename, 'rb')
        data = file.read()

        r = requests.post(
            "http://192.168.2.15:%d/asr?format=mp3" % i, data=data)

        print('%d request time %f' % (i, time.time() - t1))
        j = j + 1

       
      
        #print(r.text)
        # print(r.status_code)

    print('thread id %d ok' % i)


def main():
    run(8080)
    return
    t1 = time.time()
    li_thread = [threading.Thread(target=run, args=(i,)) for i in [8080, 8081, 8082]]
    for thread in li_thread:
        thread.start()

    while True:
        thread_running = False
        for thread in li_thread:
            if thread.is_alive():
                thread_running = True
                break
        if not thread_running:
            break
        time.sleep(0.1)

    print('time total %f' % (time.time() - t1))
    print('ok')


if __name__ == '__main__':
    main()
