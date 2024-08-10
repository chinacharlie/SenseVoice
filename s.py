from flask import Flask
from flask import request
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import librosa
import time
import random
import os
import threading
import queue
import sys
import numpy as np
import json

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH= 20*1024*1024);
app.config.update(DEBUG= True);

model_dir = "iic/SenseVoiceSmall"


model_vad = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    device="cuda:0",
)


@app.route('/pcm_asr', methods=['POST'])
def pcm_asr():
    ret = {}
    ret['code'] = 1000
    ret['message'] = 'Success'

    data = request.data
    if len(data) == 0:
        ret['code'] = 1
        ret['message'] = 'no data'
        return ret

    try:
        t1 = time.time();
        if (len(data) < 16 * 1000 * 2 * 30):
            res = model.generate(
                input=data,
                cache={},
                language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size=64)
        else:

            res = model_vad.generate(
                input=data,
                cache={},
                language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,  #
                merge_length_s=15,
            )
        
        print("pcm %fs asr %fs" % (len(data) / 32000.0, time.time() - t1))
        
    except:
        pass
    finally:
        pass

    if len(res) <= 0 or 'text' not in res[0]:
        ret['code'] = 3
        ret['message'] = 'recognize error'
        return ret

    text = rich_transcription_postprocess(res[0]["text"])

    ret['result'] = [{'text': text}]

    return json.dumps(ret, ensure_ascii=False)

    return ret


@app.route('/asr', methods=['POST'])
def asr():
    ret = {}
    ret['code'] = 1000
    ret['message'] = 'success'

    format = request.args.get('format', 'mp3')

    if len(format) == 0:
        format = 'mp3'
    if format == 'pcm':
        return pcm_asr()

    data = request.data
    if len(data) == 0:
        ret['code'] = 1
        ret['message'] = 'no data'
        return ret

    filename = 'cache/%d' % (random.random() * 1000000)
    filename = filename + '.' + format

    try:
        file = open(filename, 'wb')
        file.write(data)
    except:
        ret['code'] = 2
        ret['message'] = 'io error'
        return ret
    finally:
        file.close()

    # filename = 'cache/1 - 副本 (%d).mp3' % (random.random() * 100)

    try:
        '''
        res = model_vad.generate(
            input=filename,
            cache={},
            language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=False,  #
            merge_length_s=15,
        )
        '''
        
        res = model.generate(
                input=data,
                cache={},
                language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size=64)
    except:
        pass
    finally:
        pass

    try:
        os.remove(filename)
    except:
        pass

    if len(res) <= 0 or 'text' not in res[0]:
        ret['code'] = 3
        ret['message'] = 'recognize error'
        return ret

    text = rich_transcription_postprocess(res[0]["text"])

    ret['result'] = text

    return ret


if __name__ == '__main__':
    port = 8080
    if len(sys.argv) >= 2:
        port = int(sys.argv[1])

    app.run(debug=True, host='0.0.0.0', port=port)
