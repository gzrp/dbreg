from sanic import Sanic, json

app = Sanic("web-app")

@app.route("/")
async def hello(request):
    return json({"hello": "world"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)