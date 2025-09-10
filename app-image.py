from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image, display
from configparser import ConfigParser
import base64

config = ConfigParser()
config.read("config.ini")

# gemini-2.0-flash-exp
# gemini-1.5-flash-latest

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=config["Gemini"]["API_KEY"],
    max_tokens=8192,
)


def image4LangChain(image_url):
    if "http" in image_url:
        return {"url":image_url}
    else:
        with open(image_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        return {"url":f"data:image/jpeg;base64,{image_data}"}


user_messages = []
# append user input question
user_input = """
請根據圖片中的資訊，回傳一個JSON格式物件如下：
{
    "駕照號碼": "AB12345678",
    "姓名": "王小明",
    "性別": "男",
    "地址": "台北市信義區信義路5段123號",
    "血型": "O型",
}
如果該項目在圖片中沒有的話，就回傳N/A。
請只回傳JSON格式的物件，不要有其他文字。
"""

user_input = """
你是一名股市記者，
請根據圖片中的資訊，撰寫一篇個股報導。
並分析後續情勢。
"""

user_messages.append({"type": "text", "text": user_input + "請使用繁體中文回答。"})
# append images
# image_url = "https://i.ibb.co/KyNtMw5/IMG-20240321-172354614-AE.jpg"
# image_url = "menu.jpg"
# image_url = "https://www.taiwanhot.net/cache/985132/lgnw/medias-20211207-61af5c55e1646.jpeg"
image_url = "stock.jpg"
# image_url = "license.png"


user_messages.append(
    {
        "type": "image_url",
        "image_url": image4LangChain(image_url),
    }
)

# image_url_2 = "https://i.ibb.co/KyNtMw5/IMG-20240321-172354614-AE.jpg"
# image_url = "cat.jpg"
'''image_url_2 = "https://inv.ezpay.com.tw/images/receipt/receipt_pic.png"

user_messages.append(
    {
        "type": "image_url",
        "image_url": image4LangChain(image_url_2),
    }
)'''

human_messages = HumanMessage(content=user_messages)
result = llm.invoke([human_messages])

print("Q: " + user_input)
print("A: " + result.content)

# Display the image
display(Image(url=image_url))
# display(Image(url=image_url_2))