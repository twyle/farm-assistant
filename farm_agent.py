from dotenv import load_dotenv
load_dotenv()
import chainlit as cl
from farm_agent.agents import agent
from farm_agent.utils import load_model, evaluate_image
from PIL import Image
import io


user_location: str = None
user_name: str = None
welcome_text: str = """
Hello there. This is an application that helps farmers monitor the health level of their crops. 
Start by giving me your name and location, then upload an image of your crops. I will analyze it to 
determine the diasease or pest that affects it and then tell you how to deal with the pest or 
disease and where to purchase pesticides or fungicides.
"""

@cl.on_chat_start
async def start():
    cl.user_session.set("agent", agent)
    await cl.Message(content=welcome_text).send()
    user_name = await cl.AskUserMessage(content="What is your name?", timeout=120).send()
    user_location = await cl.AskUserMessage(content="Where are you from?", timeout=120).send()
    res = await cl.AskActionMessage(
        content="Would you like to determine if your crops are infected by a disease or by pests?",
        actions=[
            cl.Action(name="Check for diseases", value="diseases", label="✅ Check for diseases"),
            cl.Action(name="Check for Pests", value="pests", label="❌ Check for Pests")
        ]
    ).send()
    if res and res.get("value") == "diseases":
        files = None
        # Wait for the user to upload a file
        while files == None:
            files = await cl.AskFileMessage(
                content=f"{user_name['content']}, start by uploading an image of your crop.", 
                accept=["image/jpeg", "image/png", "image/jpg"]
            ).send()
        # Decode the file
        image_file = files[0]
        image_data = image_file.content # byte values of the image
        image = Image.open(io.BytesIO(image_data))
        model = load_model()
        predicted_label, predictions = evaluate_image(image, model)
        analysis_text: str = f"""
            After analyzing the image you uploaded, here is what I found:
            Maize Leaf Rust probability: {predictions['Maize Leaf Rust']}%
            Northern Leaf Blight probability: {predictions['Northern Leaf Blight']}%
            Healthy probability: {predictions['Healthy']}%
            Gray Leaf Spot probability: {predictions['Gray Leaf Spot']}%
            Your plant is most likely infected with {predicted_label}.
            """
        elements = [
            cl.Image(
                name="image2", display="inline", content=image_data
                ), 
            cl.Text(name="simple_text", content=analysis_text, display="inline", size='large')
        ]
        await cl.Message(content=f"Maize image with {predicted_label}!", elements=elements).send()
        msg = cl.Message(content="")
        await msg.send()
        await cl.sleep(1)
        msg.content = agent.run(f'Tell me some facts about the maize disease {predicted_label} especially in relation to kenya.')
        await msg.update()
        await msg.send()
        await cl.sleep(1)
        msg.content = agent.run(f'Get me aggrovets in {user_location}, Kenya')
        await msg.update()
        await cl.Message(content='Feel free to ask me more questions about maize plant diseases and how to deal with them.').send()
    else:
        await cl.Message(content='Currently cannot detect pests. Still working on that model.').send()
    

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    msg = cl.Message(content="")
    await msg.send()
    await cl.sleep(1)
    msg.content = agent.invoke({"input": message.content})["output"]
    await msg.update()