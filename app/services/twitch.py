import os
from typing import Optional

import requests


def get_user_id(username: str) -> Optional[str]:
    access_token = os.getenv("ACCESS_TOKEN")
    client_id = os.getenv("CLIENT_ID")

    # print(access_token, client_id)

    if not access_token:
        raise RuntimeError("ACCESS_TOKEN が .env に無い or 読めてないよ")
    if not client_id:
        raise RuntimeError("CLIENT_ID が .env に無い or 読めてないよ")

    url = f"https://api.twitch.tv/helix/users?login={username}"
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    # print(data)

    if "data" in data and len(data["data"]) > 0:
        user_id = data["data"][0]["id"]
        return user_id
    return None
