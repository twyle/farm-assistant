import googlemaps
import os
from langchain.agents import tool


@tool
def get_agrovets(query: str) -> str:
    """Useful when you need to get agrovets in a given location. Give it a query, such as agrovets in Nairobi, Kenya.
    """
    gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])
    results = gmaps.places(query=f'Get me aggrovets in {query}')
    aggrovet_locations: list[str] = list()
    for result in results['results']:
        bussiness: dict = dict()
        bussiness['business_status'] = result['business_status']
        bussiness['formatted_address'] = result['formatted_address']
        bussiness['name'] = result['name']
        bussiness['opening_hours'] = result.get('opening_hours', 'NaN')
        location: str = f"{result['name']}, found at {result['formatted_address']}"
        aggrovet_locations.append(location)
    return aggrovet_locations