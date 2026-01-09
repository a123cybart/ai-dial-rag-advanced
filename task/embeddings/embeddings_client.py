import json

import requests

DIAL_EMBEDDINGS = "https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings"


# TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)


class DialEmbeddingsClient:
    def __init__(self, api_key: str, deployment_name: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")
        self._api_key = api_key
        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self.headers = {"Content-Type": "application/json", "api-key": self._api_key}

    def get_embeddings(
        self, input_texts: list[str], dimensions: int
    ) -> dict[int, list[float]]:
        payload = {"input": input_texts, "dimensions": dimensions}
        response = requests.post(
            self._endpoint, headers=self.headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        response_data = response.json()

        embeddings_dict = {}
        for item in response_data.get("data", []):
            index = item["index"]
            embedding_vector = item["embedding"]
            embeddings_dict[index] = embedding_vector

        return embeddings_dict


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
