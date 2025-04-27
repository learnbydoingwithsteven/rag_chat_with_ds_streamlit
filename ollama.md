Skip to content
Navigation Menu
Product
Solutions
Resources
Open Source
Enterprise
Pricing

Search or jump to...
Sign in
Sign up
ollama
/
ollama
Public
Code
Issues
1.5k
Pull requests
248
Actions
Security
Insights
ollama/ollama
Go to file
Name		
mxyng
mxyng
model: fix build (#10416)
5cfc1c3
 · 
2 days ago
.github
llama: update to commit 71e90e88 (#10192)
2 weeks ago
api
api: fix ImageData struct comment to expect raw image bytes (#10386)
4 days ago
app
docs: improve syntax highlighting in code blocks (#8854)
2 months ago
auth
lint
9 months ago
benchmark
chore(all): replace instances of interface with any (#10067)
last month
cmd
increase default context length to 4096 (#10364)
5 days ago
convert
fixes for maverick
2 days ago
discover
chore(all): replace instances of interface with any (#10067)
last month
docs
increase default context length to 4096 (#10364)
5 days ago
envconfig
increase default context length to 4096 (#10364)
5 days ago
format
chore(all): replace instances of interface with any (#10067)
last month
fs
memory
2 days ago
integration
Integration test improvements (#9654)
2 weeks ago
kvcache
chunked attention
2 days ago
llama
llama: update to commit 2016f07b (#10352)
3 days ago
llm
explicitly decode maxarraysize 1024
2 days ago
macapp
docs: improve syntax highlighting in code blocks (#8854)
2 months ago
ml
llama4
2 days ago
model
model: fix build (#10416)
2 days ago
openai
types: include the 'items' and '$defs' fields to properly handle "arr…
3 weeks ago
parser
digest files in parallel
3 weeks ago
progress
address code review comments
2 months ago
readline
cli: adding support ctrl-n/p like general cli (#9136)
last month
runner
llama: remove model loading for grammar (#10096)
3 days ago
sample
fix test
2 days ago
scripts
Align versions for local builds (#9635)
last month
server
explicitly decode maxarraysize 1024
2 days ago
template
templates: add autotemplate for gemma3 (#9880)
last month
types
api: return model capabilities from the show endpoint (#10066)
last month
version
add version
2 years ago
.dockerignore
next build (#8539)
3 months ago
.gitattributes
chore: update gitattributes (#8860)
2 months ago
.gitignore
server/internal: copy bmizerany/ollama-go to internal package (#9294)
2 months ago
.golangci.yaml
server/internal: copy bmizerany/ollama-go to internal package (#9294)
2 months ago
CMakeLists.txt
ml/backend/ggml: use default CUDA compression mode (#10314)
2 weeks ago
CMakePresets.json
ci: silence deprecated gpu targets warning
2 days ago
CONTRIBUTING.md
CONTRIBUTING: fix code block formatting
3 weeks ago
Dockerfile
Fix dockerfile. (#9855)
3 weeks ago
LICENSE
proto -> ollama
2 years ago
Makefile.sync
llama: update to commit 2016f07b (#10352)
3 days ago
README.md
readme: add AppFlowy to community integrations (#10335)
last week
SECURITY.md
Create SECURITY.md
9 months ago
go.mod
sample: improve ollama engine sampler performance (#9374)
last month
go.sum
server/internal/registry: take over pulls from server package (#9485)
last month
main.go
lint
9 months ago
Repository files navigation
README
MIT license
Security
  ollama
Ollama
Get up and running with large language models.

macOS
Download

Windows
Download

Linux
curl -fsSL https://ollama.com/install.sh | sh
Manual install instructions

Docker
The official Ollama Docker image ollama/ollama is available on Docker Hub.

Libraries
ollama-python
ollama-js
Community
Discord
Reddit
Quickstart
To run and chat with Llama 3.2:

ollama run llama3.2
Model library
Ollama supports a list of models available on ollama.com/library

Here are some example models that can be downloaded:

Model	Parameters	Size	Download
Gemma 3	1B	815MB	ollama run gemma3:1b
Gemma 3	4B	3.3GB	ollama run gemma3
Gemma 3	12B	8.1GB	ollama run gemma3:12b
Gemma 3	27B	17GB	ollama run gemma3:27b
QwQ	32B	20GB	ollama run qwq
DeepSeek-R1	7B	4.7GB	ollama run deepseek-r1
DeepSeek-R1	671B	404GB	ollama run deepseek-r1:671b
Llama 3.3	70B	43GB	ollama run llama3.3
Llama 3.2	3B	2.0GB	ollama run llama3.2
Llama 3.2	1B	1.3GB	ollama run llama3.2:1b
Llama 3.2 Vision	11B	7.9GB	ollama run llama3.2-vision
Llama 3.2 Vision	90B	55GB	ollama run llama3.2-vision:90b
Llama 3.1	8B	4.7GB	ollama run llama3.1
Llama 3.1	405B	231GB	ollama run llama3.1:405b
Phi 4	14B	9.1GB	ollama run phi4
Phi 4 Mini	3.8B	2.5GB	ollama run phi4-mini
Mistral	7B	4.1GB	ollama run mistral
Moondream 2	1.4B	829MB	ollama run moondream
Neural Chat	7B	4.1GB	ollama run neural-chat
Starling	7B	4.1GB	ollama run starling-lm
Code Llama	7B	3.8GB	ollama run codellama
Llama 2 Uncensored	7B	3.8GB	ollama run llama2-uncensored
LLaVA	7B	4.5GB	ollama run llava
Granite-3.2	8B	4.9GB	ollama run granite3.2
Note

You should have at least 8 GB of RAM available to run the 7B models, 16 GB to run the 13B models, and 32 GB to run the 33B models.

Customize a model
Import from GGUF
Ollama supports importing GGUF models in the Modelfile:

Create a file named Modelfile, with a FROM instruction with the local filepath to the model you want to import.

FROM ./vicuna-33b.Q4_0.gguf
Create the model in Ollama

ollama create example -f Modelfile
Run the model

ollama run example
Import from Safetensors
See the guide on importing models for more information.

Customize a prompt
Models from the Ollama library can be customized with a prompt. For example, to customize the llama3.2 model:

ollama pull llama3.2
Create a Modelfile:

FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
Next, create and run the model:

ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
For more information on working with a Modelfile, see the Modelfile documentation.

CLI Reference
Create a model
ollama create is used to create a model from a Modelfile.

ollama create mymodel -f ./Modelfile
Pull a model
ollama pull llama3.2
This command can also be used to update a local model. Only the diff will be pulled.

Remove a model
ollama rm llama3.2
Copy a model
ollama cp llama3.2 my-model
Multiline input
For multiline input, you can wrap text with """:

>>> """Hello,
... world!
... """
I'm a basic program that prints the famous "Hello, world!" message to the console.
Multimodal models
ollama run llava "What's in this image? /Users/jmorgan/Desktop/smile.png"
Output: The image features a yellow smiley face, which is likely the central focus of the picture.

Pass the prompt as an argument
ollama run llama3.2 "Summarize this file: $(cat README.md)"
Output: Ollama is a lightweight, extensible framework for building and running language models on the local machine. It provides a simple API for creating, running, and managing models, as well as a library of pre-built models that can be easily used in a variety of applications.

Show model information
ollama show llama3.2
List models on your computer
ollama list
List which models are currently loaded
ollama ps
Stop a model which is currently running
ollama stop llama3.2
Start Ollama
ollama serve is used when you want to start ollama without running the desktop application.

Building
See the developer guide

Running local builds
Next, start the server:

./ollama serve
Finally, in a separate shell, run a model:

./ollama run llama3.2
REST API
Ollama has a REST API for running and managing models.

Generate a response
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt":"Why is the sky blue?"
}'
Chat with a model
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
See the API documentation for all endpoints.

Community Integrations
Web & Desktop
Open WebUI
SwiftChat (macOS with ReactNative)
Enchanted (macOS native)
Hollama
Lollms-Webui
LibreChat
Bionic GPT
HTML UI
Saddle
TagSpaces (A platform for file based apps, utilizing Ollama for the generation of tags and descriptions)
Chatbot UI
Chatbot UI v2
Typescript UI
Minimalistic React UI for Ollama Models
Ollamac
big-AGI
Cheshire Cat assistant framework
Amica
chatd
Ollama-SwiftUI
Dify.AI
MindMac
NextJS Web Interface for Ollama
Msty
Chatbox
WinForm Ollama Copilot
NextChat with Get Started Doc
Alpaca WebUI
OllamaGUI
OpenAOE
Odin Runes
LLM-X (Progressive Web App)
AnythingLLM (Docker + MacOs/Windows/Linux native app)
Ollama Basic Chat: Uses HyperDiv Reactive UI
Ollama-chats RPG
IntelliBar (AI-powered assistant for macOS)
QA-Pilot (Interactive chat tool that can leverage Ollama models for rapid understanding and navigation of GitHub code repositories)
ChatOllama (Open Source Chatbot based on Ollama with Knowledge Bases)
CRAG Ollama Chat (Simple Web Search with Corrective RAG)
RAGFlow (Open-source Retrieval-Augmented Generation engine based on deep document understanding)
StreamDeploy (LLM Application Scaffold)
chat (chat web app for teams)
Lobe Chat with Integrating Doc
Ollama RAG Chatbot (Local Chat with multiple PDFs using Ollama and RAG)
BrainSoup (Flexible native client with RAG & multi-agent automation)
macai (macOS client for Ollama, ChatGPT, and other compatible API back-ends)
RWKV-Runner (RWKV offline LLM deployment tool, also usable as a client for ChatGPT and Ollama)
Ollama Grid Search (app to evaluate and compare models)
Olpaka (User-friendly Flutter Web App for Ollama)
Casibase (An open source AI knowledge base and dialogue system combining the latest RAG, SSO, ollama support and multiple large language models.)
OllamaSpring (Ollama Client for macOS)
LLocal.in (Easy to use Electron Desktop Client for Ollama)
Shinkai Desktop (Two click install Local AI using Ollama + Files + RAG)
AiLama (A Discord User App that allows you to interact with Ollama anywhere in discord )
Ollama with Google Mesop (Mesop Chat Client implementation with Ollama)
R2R (Open-source RAG engine)
Ollama-Kis (A simple easy to use GUI with sample custom LLM for Drivers Education)
OpenGPA (Open-source offline-first Enterprise Agentic Application)
Painting Droid (Painting app with AI integrations)
Kerlig AI (AI writing assistant for macOS)
AI Studio
Sidellama (browser-based LLM client)
LLMStack (No-code multi-agent framework to build LLM agents and workflows)
BoltAI for Mac (AI Chat Client for Mac)
Harbor (Containerized LLM Toolkit with Ollama as default backend)
PyGPT (AI desktop assistant for Linux, Windows and Mac)
Alpaca (An Ollama client application for linux and macos made with GTK4 and Adwaita)
AutoGPT (AutoGPT Ollama integration)
Go-CREW (Powerful Offline RAG in Golang)
PartCAD (CAD model generation with OpenSCAD and CadQuery)
Ollama4j Web UI - Java-based Web UI for Ollama built with Vaadin, Spring Boot and Ollama4j
PyOllaMx - macOS application capable of chatting with both Ollama and Apple MLX models.
Cline - Formerly known as Claude Dev is a VSCode extension for multi-file/whole-repo coding
Cherry Studio (Desktop client with Ollama support)
ConfiChat (Lightweight, standalone, multi-platform, and privacy focused LLM chat interface with optional encryption)
Archyve (RAG-enabling document library)
crewAI with Mesop (Mesop Web Interface to run crewAI with Ollama)
Tkinter-based client (Python tkinter-based Client for Ollama)
LLMChat (Privacy focused, 100% local, intuitive all-in-one chat interface)
Local Multimodal AI Chat (Ollama-based LLM Chat with support for multiple features, including PDF RAG, voice chat, image-based interactions, and integration with OpenAI.)
ARGO (Locally download and run Ollama and Huggingface models with RAG on Mac/Windows/Linux)
OrionChat - OrionChat is a web interface for chatting with different AI providers
G1 (Prototype of using prompting strategies to improve the LLM's reasoning through o1-like reasoning chains.)
Web management (Web management page)
Promptery (desktop client for Ollama.)
Ollama App (Modern and easy-to-use multi-platform client for Ollama)
chat-ollama (a React Native client for Ollama)
SpaceLlama (Firefox and Chrome extension to quickly summarize web pages with ollama in a sidebar)
YouLama (Webapp to quickly summarize any YouTube video, supporting Invidious as well)
DualMind (Experimental app allowing two models to talk to each other in the terminal or in a web interface)
ollamarama-matrix (Ollama chatbot for the Matrix chat protocol)
ollama-chat-app (Flutter-based chat app)
Perfect Memory AI (Productivity AI assists personalized by what you have seen on your screen, heard and said in the meetings)
Hexabot (A conversational AI builder)
Reddit Rate (Search and Rate Reddit topics with a weighted summation)
OpenTalkGpt (Chrome Extension to manage open-source models supported by Ollama, create custom models, and chat with models from a user-friendly UI)
VT (A minimal multimodal AI chat app, with dynamic conversation routing. Supports local models via Ollama)
Nosia (Easy to install and use RAG platform based on Ollama)
Witsy (An AI Desktop application available for Mac/Windows/Linux)
Abbey (A configurable AI interface server with notebooks, document storage, and YouTube support)
Minima (RAG with on-premises or fully local workflow)
aidful-ollama-model-delete (User interface for simplified model cleanup)
Perplexica (An AI-powered search engine & an open-source alternative to Perplexity AI)
Ollama Chat WebUI for Docker (Support for local docker deployment, lightweight ollama webui)
AI Toolkit for Visual Studio Code (Microsoft-official VSCode extension to chat, test, evaluate models with Ollama support, and use them in your AI applications.)
MinimalNextOllamaChat (Minimal Web UI for Chat and Model Control)
Chipper AI interface for tinkerers (Ollama, Haystack RAG, Python)
ChibiChat (Kotlin-based Android app to chat with Ollama and Koboldcpp API endpoints)
LocalLLM (Minimal Web-App to run ollama models on it with a GUI)
Ollamazing (Web extension to run Ollama models)
OpenDeepResearcher-via-searxng (A Deep Research equivent endpoint with Ollama support for running locally)
AntSK (Out-of-the-box & Adaptable RAG Chatbot)
MaxKB (Ready-to-use & flexible RAG Chatbot)
yla (Web interface to freely interact with your customized models)
LangBot (LLM-based instant messaging bots platform, with Agents, RAG features, supports multiple platforms)
1Panel (Web-based Linux Server Management Tool)
AstrBot (User-friendly LLM-based multi-platform chatbot with a WebUI, supporting RAG, LLM agents, and plugins integration)
Reins (Easily tweak parameters, customize system prompts per chat, and enhance your AI experiments with reasoning model support.)
Ellama (Friendly native app to chat with an Ollama instance)
screenpipe Build agents powered by your screen history
Ollamb (Simple yet rich in features, cross-platform built with Flutter and designed for Ollama. Try the web demo.)
Writeopia (Text editor with integration with Ollama)
AppFlowy (AI collaborative workspace with Ollama, cross-platform and self-hostable)
Cloud
Google Cloud
Fly.io
Koyeb
Terminal
oterm
Ellama Emacs client
Emacs client
neollama UI client for interacting with models from within Neovim
gen.nvim
ollama.nvim
ollero.nvim
ollama-chat.nvim
ogpt.nvim
gptel Emacs client
Oatmeal
cmdh
ooo
shell-pilot(Interact with models via pure shell scripts on Linux or macOS)
tenere
llm-ollama for Datasette's LLM CLI.
typechat-cli
ShellOracle
tlm
podman-ollama
gollama
ParLlama
Ollama eBook Summary
Ollama Mixture of Experts (MOE) in 50 lines of code
vim-intelligence-bridge Simple interaction of "Ollama" with the Vim editor
x-cmd ollama
bb7
SwollamaCLI bundled with the Swollama Swift package. Demo
aichat All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI tools & agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.
PowershAI PowerShell module that brings AI to terminal on Windows, including support for Ollama
DeepShell Your self-hosted AI assistant. Interactive Shell, Files and Folders analysis.
orbiton Configuration-free text editor and IDE with support for tab completion with Ollama.
orca-cli Ollama Registry CLI Application - Browse, pull and download models from Ollama Registry in your terminal.
GGUF-to-Ollama - Importing GGUF to Ollama made easy (multiplatform)
Apple Vision Pro
SwiftChat (Cross-platform AI chat app supporting Apple Vision Pro via "Designed for iPad")
Enchanted
Database
pgai - PostgreSQL as a vector database (Create and search embeddings from Ollama models using pgvector)
Get started guide
MindsDB (Connects Ollama models with nearly 200 data platforms and apps)
chromem-go with example
Kangaroo (AI-powered SQL client and admin tool for popular databases)
Package managers
Pacman
Gentoo
Homebrew
Helm Chart
Guix channel
Nix package
Flox
Libraries
LangChain and LangChain.js with example
Firebase Genkit
crewAI
Yacana (User-friendly multi-agent framework for brainstorming and executing predetermined flows with built-in tool integration)
Spring AI with reference and example
LangChainGo with example
LangChain4j with example
LangChainRust with example
LangChain for .NET with example
LLPhant
LlamaIndex and LlamaIndexTS
LiteLLM
OllamaFarm for Go
OllamaSharp for .NET
Ollama for Ruby
Ollama-rs for Rust
Ollama-hpp for C++
Ollama4j for Java
ModelFusion Typescript Library
OllamaKit for Swift
Ollama for Dart
Ollama for Laravel
LangChainDart
Semantic Kernel - Python
Haystack
Elixir LangChain
Ollama for R - rollama
Ollama for R - ollama-r
Ollama-ex for Elixir
Ollama Connector for SAP ABAP
Testcontainers
Portkey
PromptingTools.jl with an example
LlamaScript
llm-axe (Python Toolkit for Building LLM Powered Apps)
Gollm
Gollama for Golang
Ollamaclient for Golang
High-level function abstraction in Go
Ollama PHP
Agents-Flex for Java with example
Parakeet is a GoLang library, made to simplify the development of small generative AI applications with Ollama.
Haverscript with examples
Ollama for Swift
Swollama for Swift with DocC
GoLamify
Ollama for Haskell
multi-llm-ts (A Typescript/JavaScript library allowing access to different LLM in unified API)
LlmTornado (C# library providing a unified interface for major FOSS & Commercial inference APIs)
Ollama for Zig
Abso (OpenAI-compatible TypeScript SDK for any LLM provider)
Nichey is a Python package for generating custom wikis for your research topic
Ollama for D
Mobile
SwiftChat (Lightning-fast Cross-platform AI chat app with native UI for Android, iOS and iPad)
Enchanted
Maid
Ollama App (Modern and easy-to-use multi-platform client for Ollama)
ConfiChat (Lightweight, standalone, multi-platform, and privacy focused LLM chat interface with optional encryption)
Ollama Android Chat (No need for Termux, start the Ollama service with one click on an Android device)
Reins (Easily tweak parameters, customize system prompts per chat, and enhance your AI experiments with reasoning model support.)
Extensions & Plugins
Raycast extension
Discollama (Discord bot inside the Ollama discord channel)
Continue
Vibe (Transcribe and analyze meetings with Ollama)
Obsidian Ollama plugin
Logseq Ollama plugin
NotesOllama (Apple Notes Ollama plugin)
Dagger Chatbot
Discord AI Bot
Ollama Telegram Bot
Hass Ollama Conversation
Rivet plugin
Obsidian BMO Chatbot plugin
Cliobot (Telegram bot with Ollama support)
Copilot for Obsidian plugin
Obsidian Local GPT plugin
Open Interpreter
Llama Coder (Copilot alternative using Ollama)
Ollama Copilot (Proxy that allows you to use ollama as a copilot like Github copilot)
twinny (Copilot and Copilot chat alternative using Ollama)
Wingman-AI (Copilot code and chat alternative using Ollama and Hugging Face)
Page Assist (Chrome Extension)
Plasmoid Ollama Control (KDE Plasma extension that allows you to quickly manage/control Ollama model)
AI Telegram Bot (Telegram bot using Ollama in backend)
AI ST Completion (Sublime Text 4 AI assistant plugin with Ollama support)
Discord-Ollama Chat Bot (Generalized TypeScript Discord Bot w/ Tuning Documentation)
ChatGPTBox: All in one browser extension with Integrating Tutorial
Discord AI chat/moderation bot Chat/moderation bot written in python. Uses Ollama to create personalities.
Headless Ollama (Scripts to automatically install ollama client & models on any OS for apps that depends on ollama server)
Terraform AWS Ollama & Open WebUI (A Terraform module to deploy on AWS a ready-to-use Ollama service, together with its front end Open WebUI service.)
node-red-contrib-ollama
Local AI Helper (Chrome and Firefox extensions that enable interactions with the active tab and customisable API endpoints. Includes secure storage for user prompts.)
vnc-lm (Discord bot for messaging with LLMs through Ollama and LiteLLM. Seamlessly move between local and flagship models.)
LSP-AI (Open-source language server for AI-powered functionality)
QodeAssist (AI-powered coding assistant plugin for Qt Creator)
Obsidian Quiz Generator plugin
AI Summmary Helper plugin
TextCraft (Copilot in Word alternative using Ollama)
Alfred Ollama (Alfred Workflow)
TextLLaMA A Chrome Extension that helps you write emails, correct grammar, and translate into any language
Simple-Discord-AI
LLM Telegram Bot (telegram bot, primary for RP. Oobabooga-like buttons, A1111 API integration e.t.c)
mcp-llm (MCP Server to allow LLMs to call other LLMs)
Supported backends
llama.cpp project founded by Georgi Gerganov.
Observability
Opik is an open-source platform to debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows with comprehensive tracing, automated evaluations, and production-ready dashboards. Opik supports native intergration to Ollama.
Lunary is the leading open-source LLM observability platform. It provides a variety of enterprise-grade features such as real-time analytics, prompt templates management, PII masking, and comprehensive agent tracing.
OpenLIT is an OpenTelemetry-native tool for monitoring Ollama Applications & GPUs using traces and metrics.
HoneyHive is an AI observability and evaluation platform for AI agents. Use HoneyHive to evaluate agent performance, interrogate failures, and monitor quality in production.
Langfuse is an open source LLM observability platform that enables teams to collaboratively monitor, evaluate and debug AI applications.
MLflow Tracing is an open source LLM observability tool with a convenient API to log and visualize traces, making it easy to debug and evaluate GenAI applications.
About
Get up and running with Llama 3.3, DeepSeek-R1, Phi-4, Gemma 3, Mistral Small 3.1 and other large language models.

ollama.com
Topics
go golang llama gemma mistral llm llms llava llama2 ollama qwen deepseek llama3 phi3 gemma2 phi4 gemma3
Resources
 Readme
License
 MIT license
Security policy
 Security policy
 Activity
 Custom properties
Stars
 139k stars
Watchers
 799 watching
Forks
 11.6k forks
Report repository
Releases 124
v0.6.6
Latest
2 weeks ago
+ 123 releases
Contributors
472
@mxyng
@jmorganca
@dhiltgen
@BruceMacD
@pdevine
@technovangelist
@jessegross
@bmizerany
@mchiang0610
@joshyan1
@royjhan
@ParthSareen
@rick-github
@remy415
+ 458 contributors
Languages
Go
93.8%
 
C
2.4%
 
Shell
1.1%
 
TypeScript
0.9%
 
PowerShell
0.7%
 
Inno Setup
0.4%
 
Other
0.7%
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact
Manage cookies
Do not share my personal information
