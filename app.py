from typing import List, Union, Dict
import os
from huggingface_hub import hf_hub_download, hf_hub_url
from modelscope.hub.api import HubApi
import json
import subprocess
import asyncio
import aiohttp


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


def verify_manifest(manifest):
    async def check():
        async with aiohttp.ClientSession() as session:
            for it in manifest:
                name, repo_id, files = it['name'], it['repo_id'], it['files']
                for file in files:
                    url = hf_hub_url(repo_id, filename=file)
                    async with session.head(url) as response:
                        if response.status >= 400:
                            raise ValueError(
                                f'File {file} is not found in https://huggingface.co/{repo_id}/tree/main, code {response.status}')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(check())


def main():
    manifest = read_manifest()
    verify_manifest(manifest)
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
            print(f'Move file {f} to {dist}')
            subprocess.run(['mv', f, dist])
    ms_hub.push_model(model_id='HUODONG/stable-diffusion-pack',
                      model_dir=os.path.join(os.getcwd(), 'store'),
                      )


if __name__ == '__main__':
    main()
