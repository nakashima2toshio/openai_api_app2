# util_02_images.py
from openai import OpenAI
from openai.types.responses import response
import base64

# -----------------------------------------
#（1-1）入力画像（URL）からテキストを作成する。
# -----------------------------------------
def client_responses_create_image_api(model, input_text, image_url) -> response:
    client = OpenAI()
    res_text = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": input_text},
                {
                    "type": "input_image",
                    "image_url": image_url,
                },
            ],
        }],
    )
    return res_text

# -----------------------------------------
# (1-2) 入力画像データ（Base64）からテキストを作成する。
# -----------------------------------------
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# (1-2) 入力画像データ（Base64)からテキストデータを作成する。

# (2) プロンプトから画像イメージのジェネレート：Generate an image
def client_images_generate_api(prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1) -> response:
    from openai import OpenAI

    client = OpenAI()
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url


def client_response_create_images_api():
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "what is in this image?" },
                    {
                        "type": "input_image",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                ]
            }
        ]
    )
    return response


def main():
    # （1-1）入力画像（URL）からテキストを作成する。
    model = 'gpt-4o-mini'
    user_content_text = "what's in this image? 日本語で"
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg'
    # image_url = 'https://drive.google.com/file/d/14Yw1qCLRiCVwdahw9jmKXekflUpqRICK/view?usp=drive_link'
    res = client_responses_create_image_api(model, user_content_text, image_url)
    print(res)

if __name__ == "__main__":
    main()

