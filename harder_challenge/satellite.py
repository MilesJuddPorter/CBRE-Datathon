from dotenv import find_dotenv, load_dotenv
from io import BytesIO 
from PIL import Image
import requests
import geopy
import os


load_dotenv(find_dotenv())


class Satellite:
    def __init__(self):

        # API KEYS
        self.BING_API_KEY = os.environ['BING_API_KEY']
        self.GOOGLE_MAPS_API_KEY = os.environ['GOOGLE_MAPS_API_KEY']

        # Clients
        self.geolocator = geopy.geocoders.Bing(self.BING_API_KEY)

    def address_to_coordinates(self, address, formattingType="Bing"):
        """
        Takes in an address and returns coordinates

        Parameters
        ----------
        address : str, required
            Address of location to get coords for
        formattingType : str, optional
            Defines the type of output format ['Bing', 'Google'] (default is 'Bing')

        Returns
        -------
        str
            Formatted latitude and longitude as 'lat,lng'
        """
        if formattingType == "Bing":
            location = self.geolocator.geocode(address)
            if hasattr(location, 'latitude') and hasattr(location, 'longitude'):
                return f"{location.latitude},{location.longitude}"
            else:
                return None
        else:
            return "Only 'Bing' formattingType is active"

    def bing_metadata(self, coords,
                    imagerySet="BirdseyeV2",
                    orientation=0,
                    zoomLevel=None):
        """
        Gets metadata information before making image request
        
        Parameters
        ----------
        coords : str, required
            Format as string like 'latitude,longitude'  
        imagerySet : str, optional
            Image Set from Bing API (default is BirdseyeV2)
        orientation : str, optional
            Not currently used in imagery (Inactive)
        zoomLevel : int, optional
            zoomLevel for metadata (default is None)

        Returns
        -------
        dict
            Metadata dictionary
        """

        try:
            if zoomLevel:
                URL = f"https://dev.virtualearth.net/REST/v1/Imagery/BasicMetadata/{imagerySet}/{coords}?&zoomLevel={zoomLevel}&key={self.BING_API_KEY}"
            else:
                URL = f"https://dev.virtualearth.net/REST/v1/Imagery/BasicMetadata/{imagerySet}/{coords}?&key={self.BING_API_KEY}"
            resp = requests.get(URL)
            data = resp.json()

            return data
        except:
            return None

    def bing_imagery(self, coords,
                    imagerySet="BirdseyeV2",
                    orientation=0,
                    zoomLevel=10):
        """
        Gets metadata information before making image request
        
        Parameters
        ----------
        coords : str, required
            Format as string like 'latitude,longitude'  
        imagerySet : str, optional
            Image Set from Bing API (default is BirdseyeV2)
        orientation : str, optional
            Not currently used in imagery (Inactive)
        zoomLevel : str, optional
            Not used so as to give back information to request zoomLevel on image

        Returns
        -------
        PIL.Image
            Requested image
        """

        try:
            URL = f"https://dev.virtualearth.net/REST/v1/Imagery/Map/{imagerySet}/{coords}/{zoomLevel}?&key={self.BING_API_KEY}"
            resp = requests.get(URL)
            img_bytes = resp.content
            image = Image.open(BytesIO(img_bytes))

            return image
        except Exception as e:
            print(f"Error on image call: {str(e)}")
            return resp

    def bing_pipeline(self, address, imagerySet, zoomLevel=None):
        """
        Creates a pipeline for processing Bing image

        Parameters
        ----------
        address : str, required
            Address of location to get coords for
        imagerySet : str, required
            Image Set from Bing API

        Returns
        ------
        touple
            image : PIL.Image
                Image returned from the bing request
            metadata : dict
                Metadata found for the bing request

        """

        # Get Coords
        coords = self.address_to_coordinates(address, formattingType="Bing")
        if not coords:
            raise Exception(f"Could not find coords for address: {address}")
        
        # Get metadata
        metadata = self.bing_metadata(coords, imagerySet=imagerySet, zoomLevel=zoomLevel)
        if not metadata:
            raise Exception(f"Could not find Metadata for coords: {coords}")
        elif metadata.get("statusCode", 400) == 400:
            raise Exception(f"[ERROR] - Metadata request\nReceived a 400 status code given: {coords}\n\nFull Response: {metadata}")
        
        print(f"METADATA: {metadata}")

        # Get Image
        if not zoomLevel:
            zoomLevel = (metadata['resourceSets'][0]['resources'][0]['zoomMax'] + metadata['resourceSets'][0]['resources'][0]['zoomMin']) // 2

        image = self.bing_imagery(coords, imagerySet=imagerySet, zoomLevel=zoomLevel)

        return image, metadata