from typing import Union, Dict
import os
from huggingface_hub import hf_hub_download
from modelscope.hub.api import HubApi
import json
import subprocess


def read_manifest():
    with (open('manifest.json', 'r')) as f:
        return json.load(f)


def modelscope_hub() -> HubApi:
    token = os.environ.get('MODELSCOPE_TOKEN')
    if token == None:
        raise ValueError(
            'Please set the MODELSCOPE_TOKEN environment variable')
    hub = HubApi()
    hub.login(token)
    return hub


def main():
    manifest = read_manifest()
    ms_hub = modelscope_hub()
    for it in manifest:
        name, repo_id, files = it['name'], it['repo_id'], it['files']
        print('Downloading', name)
        for file in files:
            f = hf_hub_download(repo_id, filename=file)
            dist = os.path.basename(os.path.join(
                os.getcwd(), 'store', repo_id, file))
            if not os.path.exists(dist):
                os.makedirs(dist, exist_ok=True)
            subprocess.run(['cp', f, dist])

    ms_hub.push_model(model_id='HUODONG/stable-diffusion-pack',
                      model_dir=os.path.join(os.getcwd(), 'store'))


if __name__ == '__main__':
    main()
