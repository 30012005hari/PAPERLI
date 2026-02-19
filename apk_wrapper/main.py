"""
Android APK Wrapper for Research Paper Explainer.

This is a Kivy WebView app that wraps the Streamlit web app.
The app connects to a remote Streamlit server URL.

Prerequisites:
- Install Buildozer: pip install buildozer
- On Linux/WSL: sudo apt install -y build-essential libffi-dev python3-dev
- Set your server URL below

Build:
    buildozer android debug
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.clock import Clock
import webbrowser

# ── Change this to your server's URL ──
DEFAULT_SERVER_URL = "http://localhost:8502"


class ExplainerApp(App):
    """Android wrapper that opens the web app in the system browser."""

    def build(self):
        self.title = "Research Paper Explainer"

        layout = BoxLayout(orientation="vertical", padding=40, spacing=20)

        # Title
        title = Label(
            text="🧬 Research Paper\nExplainer",
            font_size="28sp",
            halign="center",
            size_hint_y=0.3,
            bold=True,
        )
        title.bind(size=title.setter('text_size'))
        layout.add_widget(title)

        # Server URL input
        url_label = Label(
            text="Server URL:",
            font_size="16sp",
            size_hint_y=0.1,
            halign="left",
        )
        layout.add_widget(url_label)

        self.url_input = TextInput(
            text=DEFAULT_SERVER_URL,
            font_size="16sp",
            size_hint_y=0.1,
            multiline=False,
        )
        layout.add_widget(self.url_input)

        # Open button
        open_btn = Button(
            text="🚀 Open App",
            font_size="20sp",
            size_hint_y=0.15,
            background_color=(0.39, 0.40, 0.95, 1),
        )
        open_btn.bind(on_press=self.open_app)
        layout.add_widget(open_btn)

        # Status
        self.status = Label(
            text="Enter your server URL and tap Open",
            font_size="14sp",
            size_hint_y=0.1,
            color=(0.6, 0.6, 0.6, 1),
        )
        layout.add_widget(self.status)

        # Spacer
        layout.add_widget(Label(size_hint_y=0.25))

        return layout

    def open_app(self, instance):
        url = self.url_input.text.strip()
        if url:
            self.status.text = f"Opening {url}..."
            webbrowser.open(url)
        else:
            self.status.text = "Please enter a valid URL"


if __name__ == "__main__":
    ExplainerApp().run()
