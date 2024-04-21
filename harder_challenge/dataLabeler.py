import base64
import requests
from dotenv import load_dotenv, find_dotenv
import os


class DataLabeler:
    def __init__(self):
       # Ensure the .env file is in the root of the project
       load_dotenv(find_dotenv())
       self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def encode_image(self, image_path):
        """
        Encodes an image to base64
        
        Parameters
        ----------
        image_path : str, required
            Path to the image to encode
        """

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def label_image(self, image_path):
        """
        Labels an image based on the categories provided
        
        Parameters
        ----------
        image_path : str, required
            Path to the image to label
        """

        prefix = """
        This image is a satellite image view of an address. Please label the image based on the following categories:
        The hardest categories to define accurately are 2, 3, and 4. 1 and 5 are usually easy to define.
        Spend extra time on 2, 3, and 4. Attention to detail is critical, follow the category information provided as close as possible.
        """

        categories = """
            1:
            - Undeveloped land
            - Free of structures
            - No sign of land having been cleared or bulldozed
            - Vegetation or sand might appear to be unmoved

            2:
            - Ground Broken
            - Free of structures
            - Discernable movement of land with possible temporary roads
            - May be construction vehicles present
            - Vegetation cleared

            3:
            - Concrete Pad:
            - Land contains a large flat surface of poured concrete
            - The land around that pad is still unfinished surface usually dirt and free of vegetation
            - May be surrounded by construction vehicles
            
            4:
            - Framing Going up:
            - Contains many of the same elements of concrete pad
            - Instead of a large flat surface you begin to see more variation in pixel color as the complexity of
            the structure begins to take shape
            - Walls may be casting shadows depending on the time of day of the satellite photo.
            - There may be a roof present but there is no paved parking lot present.
            
            5:
            - Near completion or completed:
            - A polished appearance.
            - Structure has a roof (often white but not always
            - Freshly paved parking lot. May have vehicles in the lot or trailers (containers) backed up along at
            least one side of the building.
            - May have attractive plantings to demarcate portions of the parking lot.
            """
        
        format = "Reply with just the number of the correct category. For example, if the correct category is 1, reply with 1. No other text is allowed"
        

        message = f"""
        {prefix}
        {categories}
        Follow this output:
        {format}
        """
        base64_image = self.encode_image(image_path)

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.openai_api_key}"
        }

        payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": message
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        label = response.json()['choices'][0]['message']['content']
        return label