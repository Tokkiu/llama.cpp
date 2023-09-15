test -f Application-Installer.dmg && rm Application-Installer.dmg
create-dmg \
  --volname "LLM server Installer" \
  --volicon "llm_icon.icns" \
  --background "llm_background.png" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "llama.app" 200 190 \
  --hide-extension "llama.app" \
  --app-drop-link 600 185 \
  "llama.dmg" \
  "dist/llama.app/"