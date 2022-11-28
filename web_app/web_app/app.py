from click_web import create_click_web_app

from web_app import commands

app = create_click_web_app(commands, commands.cli)
