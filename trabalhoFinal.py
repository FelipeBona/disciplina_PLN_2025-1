import json
import logging
import os
import random
import re
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import scrolledtext, messagebox, ttk, filedialog
import requests
from collections import defaultdict
from string import punctuation

# Configuração de logging
logging.basicConfig(
    filename='gpteco_chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)


class GPTEcoChatbot:
    def __init__(self):
        # Banco de dados em memória expandido
        self.knowledge_base = {
            r'oi|olá|hello|eae|bom dia|boa tarde|boa noite': [
                'Olá! Sou o GPTEco, seu assistente virtual. Como posso ajudar você hoje?',
                'Oi! Tudo bem? Estou pronto para responder suas perguntas!',
                'Saudações! Em que posso te ajudar agora?'
            ],
            r'quem é você|quem és tu|qual seu nome': [
                'Eu sou o GPTEco, um assistente virtual criado para fornecer respostas úteis e precisas!',
                'GPTEco, seu parceiro para dúvidas de todos os tipos. Qual é a sua?',
                'Sou o GPTEco, uma IA desenvolvida para ajudar e informar!'
            ],
            r'como você está|está bem|tudo bem': [
                'Estou 100% operacional e pronto para ajudar! E você, como está?',
                'Tudo ótimo por aqui! Qual é a sua vibe hoje?',
                'Como uma IA, estou sempre bem. E tu, tá de boa?'
            ],
            r'tchau|adeus|até logo|encerrar|sair': [
                'Até mais! Foi ótimo conversar com você!',
                'Tchau! Volte sempre que precisar!',
                'Até a próxima! Estou sempre aqui.'
            ],
            r'o que é python|python': [
                'Python é uma linguagem de programação versátil, usada em desenvolvimento web, IA, automação e mais. Quer saber sobre algum projeto em Python?',
                'Python é uma linguagem interpretada de alto nível, conhecida por sua simplicidade e poder. Já experimentou programar com ela?',
                'É uma linguagem de programação super popular, ideal para iniciantes e experts. Posso te ajudar com algum código?'
            ],
            r'qual é a capital do brasil|capital do brasil': [
                'A capital do Brasil é Brasília, planejada por Oscar Niemeyer e Lúcio Costa, inaugurada em 1960.',
                'Brasília é a capital do Brasil, conhecida por sua arquitetura moderna e urbanismo único!',
                'É Brasília! Quer saber mais sobre a história da cidade?'
            ],
            r'qual é a capital da frança|capital da frança': [
                'A capital da França é Paris, famosa pela Torre Eiffel, Louvre e sua cultura vibrante.',
                'Paris, a cidade do amor e da luz, é a capital da França!',
                'É Paris! Quer dicas sobre o que fazer lá?'
            ],
            r'o que é inteligência artificial|o que é ia|inteligência artificial': [
                'Inteligência Artificial (IA) é o campo da computação que cria sistemas capazes de aprender, raciocinar e tomar decisões, simulando habilidades humanas.',
                'IA é a tecnologia por trás de assistentes como eu, permitindo que máquinas processem dados e respondam de forma inteligente!',
                'É a ciência de fazer máquinas pensarem como humanos. Quer exemplos de aplicações de IA?'
            ],
            r'você é inteligente|é esperto': [
                'Digamos que sou esperto o suficiente para ajudar com suas perguntas! Minha inteligência vem de dados e algoritmos bem projetados.',
                'Sou programado para ser útil, então posso parecer bem esperto às vezes! Quer me testar?',
                'Minha "inteligência" é mais sobre encontrar as melhores respostas para você. Qual é a sua dúvida?'
            ],
            r'obrigado|obrigada|valeu|agradeço': [
                'De nada! Sempre um prazer ajudar!',
                'Valeu pelo carinho! Estou aqui para mais.',
                'Por nada! Qual é a próxima?'
            ],
            r'qual é o sentido da vida': [
                'Segundo O Guia do Mochileiro das Galáxias, é 42! Mas, falando sério, o sentido da vida pode ser encontrar o que te faz feliz. Qual é o seu sentido?',
                'Essa é profunda! Para alguns, é amor, conhecimento ou aventura. Qual é a sua visão?',
                'Boa pergunta! Talvez seja aprender, crescer e ajudar os outros. O que você acha?'
            ],
            r'quem criou você|quem te fez': [
                'Fui criado por uma equipe de desenvolvedores apaixonados por IA, com um toque de criatividade e tecnologia!',
                'Sou uma criação de mentes brilhantes que queriam um assistente virtual útil. Aqui estou eu!',
                'Meus criadores são experts em IA, e eu sou o resultado do trabalho deles para ajudar usuários como você!'
            ],
            r'como aprender programação|aprender a programar': [
                'Programação é incrível! Comece com linguagens como Python ou JavaScript. Sites como Codecademy, freeCodeCamp e Coursera são ótimos. Quer dicas para iniciantes?',
                'Recomendo escolher uma linguagem (ex.: Python), praticar com projetos pequenos e usar recursos como YouTube ou livros como "Automate the Boring Stuff". Qual área te interessa?',
                'Aprender a programar é sobre prática! Tente resolver problemas no LeetCode ou HackerRank. Posso sugerir um plano de estudos!'
            ],
            r'qual é a melhor linguagem de programação': [
                'Não há "melhor" linguagem, depende do objetivo! Python é ótimo para IA e iniciantes, JavaScript para web, C++ para desempenho. Qual é o seu projeto?',
                'Python é super versátil, mas JavaScript domina a web e C é ideal para sistemas. Me conta o que você quer fazer!',
                'Cada linguagem tem seu brilho. Quer criar apps? Tente Java ou Kotlin. Web? JavaScript. IA? Python. Qual é a sua meta?'
            ],
            r'tempo|clima|previsão do tempo': [
                'Não tenho acesso a dados de clima em tempo real, mas posso buscar previsões gerais. Me diz a cidade e eu pesquiso!',
                'Quer saber o clima? Informe a cidade, e eu busco informações atualizadas na web!',
                'Clima é sempre uma boa conversa! Qual cidade você quer checar?'
            ],
            r'piada|conte uma piada': [
                'Por que o astronauta terminou com a namorada? Porque ele precisava de espaço!',
                'O que o zero disse pro oito? Que cinto maneiro!',
                'Por que o programador prefere o modo escuro? Porque a luz atrai bugs.'
            ],
            r'o que é machine learning|machine learning': [
                'Machine Learning é um ramo da IA onde máquinas aprendem padrões a partir de dados para fazer previsões ou decisões. Exemplo: recomendação de filmes na Netflix!',
                'É quando computadores "aprendem" com dados, como prever preços ou reconhecer imagens. Quer exemplos práticos?',
                'Um subcampo da IA que usa algoritmos para aprender com dados. Quer saber sobre redes neurais ou outros tópicos?'
            ],
            r'qual é a maior cidade do mundo': [
                'Tóquio, no Japão, é considerada a maior cidade do mundo em população, com cerca de 37 milhões de habitantes na área metropolitana!',
                'A maior cidade em termos de população é Tóquio. Quer saber mais sobre urbanismo ou outras cidades grandes?',
                'Tóquio lidera com milhões de pessoas. Curioso sobre outras megacidades?'
            ]
        }

        self.default_responses = [
            "Hmm, essa é nova para mim! Posso pesquisar na web ou você pode me ensinar a resposta. O que prefere?",
            "Não tenho essa informação, mas posso buscar online ou aprender com você! Qual é a próxima?",
            "Boa pergunta! Não sei ainda, mas posso pesquisar ou você me ensina. Topa?",
            "Ainda não aprendi isso. Quer que eu busque na internet ou prefere me ensinar?",
            "Ops, essa passou! Me ensina a resposta ou quer que eu pesquise?"
        ]

        self.conversation_history = []
        self.context = defaultdict(list)  # Armazena contexto por tópicos
        self.bot_name = "GPTEco"
        self.version = "3.2 NLP Enhanced"
        self.last_update = "2025-05-27"
        self.knowledge_file = "knowledge_base.json"
        self.load_knowledge_base()

        # Dicionário para análise de sentimento
        self.sentiment_dict = {
            'positive': ['ótimo', 'bom', 'excelente', 'feliz', 'maravilhoso', 'gostei', 'adorei'],
            'negative': ['ruim', 'triste', 'péssimo', 'chato', 'decepcionado', 'horrível'],
            'neutral': ['ok', 'tá', 'normal', 'bem']
        }

        # Lista de entidades comuns para NER
        self.entity_types = {
            'location': ['brasil', 'frança', 'paris', 'brasília', 'tóquio', 'são paulo', 'rio de janeiro'],
            'person': ['oscar niemeyer', 'lúcio costa'],
            'date': [r'\d{1,2}/\d{1,2}/\d{4}', r'\d{1,2} de [a-z]+ de \d{4}']
        }

    def load_knowledge_base(self):
        """Carrega a base de conhecimento de um arquivo JSON, se existir"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    loaded_knowledge = json.load(f)
                self.knowledge_base.update(loaded_knowledge)
                logging.info("Base de conhecimento carregada de knowledge_base.json")
        except Exception as e:
            logging.error(f"Erro ao carregar base de conhecimento: {str(e)}")

    def save_knowledge_base(self):
        """Atualiza a base de conhecimento em um único arquivo JSON"""
        try:
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=4)
            logging.info("Base de conhecimento atualizada em knowledge_base.json")
            return True
        except Exception as e:
            logging.error(f"Erro ao atualizar base de conhecimento: {str(e)}")
            return False

    def add_knowledge(self, pattern, response):
        """Adiciona uma nova pergunta/resposta ao banco de dados"""
        try:
            if not pattern or not response:
                raise ValueError("Padrão e resposta não podem estar vazios")

            pattern = pattern.lower().strip()
            response = response.strip()

            # Verifica se o padrão regex é válido
            try:
                re.compile(pattern)
            except re.error:
                raise ValueError("Padrão regex inválido")

            # Adiciona ou atualiza o padrão
            if pattern in self.knowledge_base:
                self.knowledge_base[pattern].append(response)
            else:
                self.knowledge_base[pattern] = [response]

            # Atualiza o arquivo JSON
            self.save_knowledge_base()
            logging.info(f"Novo conhecimento adicionado: {pattern} -> {response}")
            return True

        except Exception as e:
            logging.error(f"Erro ao adicionar conhecimento: {str(e)}")
            return False

    def tokenize(self, text):
        """Tokeniza o texto em palavras, removendo pontuação"""
        text = text.lower().strip()
        text = ''.join(char for char in text if char not in punctuation)
        return text.split()

    def extract_keywords(self, user_input):
        """Extrai palavras-chave do texto do usuário"""
        tokens = self.tokenize(user_input)
        # Remove stop words comuns
        stop_words = {'é', 'a', 'o', 'em', 'de', 'para', 'com', 'que', 'e', 'um', 'uma'}
        keywords = [token for token in tokens if token not in stop_words]
        return keywords

    def analyze_sentiment(self, user_input):
        """Analisa o sentimento da mensagem do usuário"""
        tokens = self.tokenize(user_input)
        score = 0
        for token in tokens:
            if token in self.sentiment_dict['positive']:
                score += 1
            elif token in self.sentiment_dict['negative']:
                score -= 1
            elif token in self.sentiment_dict['neutral']:
                score += 0.5
        if score > 0:
            return 'positive'
        elif score < 0:
            return 'negative'
        return 'neutral'

    def extract_entities(self, user_input):
        """Extrai entidades nomeadas (locais, pessoas, datas)"""
        entities = {'location': [], 'person': [], 'date': []}
        user_input = user_input.lower()

        # Verifica entidades de localização e pessoas
        for entity_type, entity_list in self.entity_types.items():
            if entity_type in ['location', 'person']:
                for entity in entity_list:
                    if entity in user_input:
                        entities[entity_type].append(entity)
            elif entity_type == 'date':
                for pattern in entity_list:
                    matches = re.findall(pattern, user_input)
                    entities['date'].extend(matches)

        return entities

    def update_context(self, user_input, keywords, entities):
        """Atualiza o contexto da conversa com base em palavras-chave e entidades"""
        for keyword in keywords:
            self.context[keyword].append(user_input)
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                self.context[entity].append(user_input)

    def get_contextual_response(self, user_input, base_response):
        """Adapta a resposta com base no contexto e sentimento"""
        sentiment = self.analyze_sentiment(user_input)
        entities = self.extract_entities(user_input)

        # Ajusta a resposta com base no sentimento
        if sentiment == 'positive':
            prefix = random.choice([
                "Que bom te ver tão animado! ",
                "Adoro sua energia positiva! ",
                "Que vibe boa! "
            ])
        elif sentiment == 'negative':
            prefix = random.choice([
                "Parece que você está meio pra baixo, posso ajudar? ",
                "Tudo bem? Vamos tentar melhorar seu dia! ",
                "Sinto que você não está tão animado, quer conversar? "
            ])
        else:
            prefix = ""

        # Inclui entidades na resposta, se aplicável
        if entities['location'] or entities['person'] or entities['date']:
            entity_text = []
            if entities['location']:
                entity_text.append(f"Você mencionou {', '.join(entities['location'])}")
            if entities['person']:
                entity_text.append(f"e falou sobre {', '.join(entities['person'])}")
            if entities['date']:
                entity_text.append(f"em relação a {', '.join(entities['date'])}")
            entity_suffix = f" ({', '.join(entity_text)})."
        else:
            entity_suffix = ""

        return f"{prefix}{base_response}{entity_suffix}"

    def find_response(self, user_input):
        """Busca a melhor resposta na base de conhecimento com NLP"""
        try:
            user_input = user_input.lower().strip()
            if not user_input:
                raise ValueError("Entrada vazia não permitida")

            # Extrai palavras-chave e entidades
            keywords = self.extract_keywords(user_input)
            entities = self.extract_entities(user_input)
            self.update_context(user_input, keywords, entities)

            best_response = None
            best_score = 0

            # Calcula pontuação com regex e similaridade de palavras-chave
            for pattern, responses in self.knowledge_base.items():
                matches = len(re.findall(pattern, user_input))
                pattern_tokens = set(self.tokenize(pattern.replace('|', ' ')))
                input_tokens = set(keywords)
                common_tokens = pattern_tokens & input_tokens
                token_score = len(common_tokens) / max(len(pattern_tokens), 1)

                score = matches * (1 if re.fullmatch(pattern, user_input) else 0.5) + token_score
                if score > best_score:
                    best_score = score
                    best_response = random.choice(responses)

            if best_response:
                return self.get_contextual_response(user_input, best_response)
            return None

        except Exception as e:
            logging.error(f"Erro ao processar entrada: {str(e)}")
            return None

    def search_web(self, query):
        """Realiza uma pesquisa na web usando a API do DuckDuckGo com formatação avançada"""
        try:
            if not query or len(query) < 3:
                return random.choice(self.default_responses)

            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("AbstractText"):
                abstract = data["AbstractText"][:300] + ("..." if len(data["AbstractText"]) > 300 else "")
                return self.get_contextual_response(query, f"🔍 De acordo com minha pesquisa: {abstract}")
            elif data.get("RelatedTopics"):
                for topic in data["RelatedTopics"]:
                    if "Text" in topic:
                        text = topic['Text'][:300] + ("..." if len(topic['Text']) > 300 else "")
                        return self.get_contextual_response(query, f"🔍 Encontrei esta informação: {text}")
            elif data.get("AbstractURL"):
                return self.get_contextual_response(query,
                                                    f"ℹ️ Não tenho uma resposta direta, mas você pode encontrar mais informações aqui: {data['AbstractURL']}")

            return random.choice(self.default_responses)

        except requests.exceptions.RequestException as e:
            logging.error(f"Erro na pesquisa web: {str(e)}")
            return "⚠️ Não consegui conectar à internet. Tente novamente ou me ensine a resposta!"
        except Exception as e:
            logging.error(f"Erro inesperado na pesquisa: {str(e)}")
            return random.choice(self.default_responses)

    def save_conversation(self, format="json", silent=False):
        """Salva o histórico da conversa em JSON ou TXT"""
        try:
            if not self.conversation_history:
                return False, "Nenhuma conversa para salvar"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'conversa_gpteco_{timestamp}.{format}'

            if format == "json":
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": {
                            "bot_name": self.bot_name,
                            "version": self.version,
                            "saved_at": datetime.now().isoformat()
                        },
                        "conversation": self.conversation_history
                    }, f, ensure_ascii=False, indent=4)
            elif format == "txt":
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Histórico de conversa com {self.bot_name} (v{self.version})\n")
                    f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    for entry in self.conversation_history:
                        if "user" in entry:
                            f.write(f"👤 Você ({entry['time']}): {entry['user']}\n")
                        else:
                            f.write(f"🤖 {self.bot_name} ({entry['time']}): {entry['bot']}\n")

            logging.info(f"Histórico salvo em {filename}")
            return True, filename

        except Exception as e:
            logging.error(f"Erro ao salvar conversa: {str(e)}")
            return False, str(e)

    def load_conversation(self, filename):
        """Carrega uma conversa de um arquivo JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "conversation" not in data:
                raise ValueError("Arquivo JSON inválido: campo 'conversation' não encontrado")

            return data["conversation"]

        except Exception as e:
            logging.error(f"Erro ao carregar conversa: {str(e)}")
            return None


class GPTEcoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"GPTEco Chatbot - Assistente Virtual Inteligente v3.2")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        self.center_window()

        self.chatbot = GPTEcoChatbot()
        self.theme = "light"
        self.sidebar_visible = True
        self.status_message = tk.StringVar()
        self.status_message.set("Bem-vindo ao GPTEco! Digite sua pergunta abaixo.")
        self.setup_styles()
        self.create_widgets()

        # Configuração de atalhos
        self.root.bind("<Control-l>", lambda event: self.clear_chat())
        self.root.bind("<Control-h>", lambda event: self.show_help())
        self.root.bind("<Control-t>", lambda event: self.toggle_theme())
        self.root.bind("<Control-s>", lambda event: self.save_conversation("json"))
        self.root.bind("<Control-o>", lambda event: self.load_conversation_from_file())
        self.root.bind("<Control-p>", lambda event: self.show_lgpd())
        self.root.bind("<Control-n>", lambda event: self.show_nlp_info())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Exibir mensagem inicial
        self.display_welcome_message()

    def center_window(self):
        """Centraliza a janela na tela"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def display_welcome_message(self):
        """Exibe mensagem de boas-vindas"""
        welcome_msg = (
            f"🤖 {self.chatbot.bot_name} (v{self.chatbot.version}): "
            f"Bem-vindo ao GPTEco, seu assistente virtual com Processamento de Linguagem Natural!\n\n"
            f"Estou equipado com análise de sentimento, extração de palavras-chave, reconhecimento de entidades e contexto de conversa.\n"
            f"Use a barra lateral para explorar funcionalidades ou pressione Ctrl+N para aprender sobre NLP.\n\n"
            f"🔹 Exemplos de perguntas:\n"
            f"- Qual é a capital do Brasil?\n"
            f"- Estou feliz, conte uma piada!\n"
            f"- O que Oscar Niemeyer fez em Brasília?\n"
            f"- Como aprender programação?\n\n"
            f"📌 Dicas:\n"
            f"- Pressione Ctrl+H para ajuda\n"
            f"- Use Ctrl+N para informações sobre NLP\n"
            f"- Ensine-me algo novo com o botão 'Ensinar'!\n"
        )
        self.display_message(welcome_msg, "bot")

    def setup_styles(self):
        """ConfigConcurrentModificationExceptionura estilos visuais para o tema claro/escuro"""
        self.style = ttk.Style()

        self.colors = {
            "light": {
                "primary_bg": "#fafafa",
                "secondary_bg": "#ffffff",
                "text": "#212121",
                "accent": "#1976d2",
                "user_bubble": "#bbdefb",
                "bot_bubble": "#e0e0e0",
                "button_bg": "#1976d2",
                "button_fg": "#ffffff",
                "input_bg": "#ffffff",
                "highlight": "#90caf9",
                "sidebar_bg": "#eceff1",
                "border": "#bdbdbd",
                "status_bg": "#f5f5f5",
                "status_fg": "#424242"
            },
            "dark": {
                "primary_bg": "#212121",
                "secondary_bg": "#2d2d2d",
                "text": "#e0e0e0",
                "accent": "#42a5f5",
                "user_bubble": "#0288d1",
                "bot_bubble": "#424242",
                "button_bg": "#42a5f5",
                "button_fg": "#ffffff",
                "input_bg": "#424242",
                "highlight": "#0288d1",
                "sidebar_bg": "#2d2d2d",
                "border": "#616161",
                "status_bg": "#37474f",
                "status_fg": "#b0bec5"
            }
        }

        # Tema claro
        self.style.theme_create("light", parent="alt", settings={
            "TFrame": {"configure": {"background": self.colors["light"]["primary_bg"]}},
            "TLabel": {"configure": {
                "background": self.colors["light"]["primary_bg"],
                "foreground": self.colors["light"]["text"],
                "font": ("Helvetica", 10)
            }},
            "TButton": {"configure": {
                "background": self.colors["light"]["button_bg"],
                "foreground": self.colors["light"]["button_fg"],
                "font": ("Helvetica", 10, "bold"),
                "padding": 8,
                "borderwidth": 0,
                "relief": "flat"
            }, "map": {
                "background": [("active", self.colors["light"]["highlight"])]
            }},
            "TEntry": {"configure": {
                "fieldbackground": self.colors["light"]["input_bg"],
                "foreground": self.colors["light"]["text"],
                "font": ("Helvetica", 11),
                "insertcolor": self.colors["light"]["text"],
                "borderwidth": 1,
                "relief": "flat"
            }},
            "Vertical.TScrollbar": {"configure": {
                "background": self.colors["light"]["button_bg"],
                "troughcolor": self.colors["light"]["primary_bg"],
                "arrowcolor": self.colors["light"]["button_fg"]
            }},
            "Status.TLabel": {"configure": {
                "background": self.colors["light"]["status_bg"],
                "foreground": self.colors["light"]["status_fg"],
                "font": ("Helvetica", 9)
            }}
        })

        # Tema escuro
        self.style.theme_create("dark", parent="alt", settings={
            "TFrame": {"configure": {"background": self.colors["dark"]["primary_bg"]}},
            "TLabel": {"configure": {
                "background": self.colors["dark"]["primary_bg"],
                "foreground": self.colors["dark"]["text"],
                "font": ("Helvetica", 10)
            }},
            "TButton": {"configure": {
                "background": self.colors["dark"]["button_bg"],
                "foreground": self.colors["dark"]["button_fg"],
                "font": ("Helvetica", 10, "bold"),
                "padding": 8,
                "borderwidth": 0,
                "relief": "flat"
            }, "map": {
                "background": [("active", self.colors["dark"]["highlight"])]
            }},
            "TEntry": {"configure": {
                "fieldbackground": self.colors["dark"]["input_bg"],
                "foreground": self.colors["dark"]["text"],
                "font": ("Helvetica", 11),
                "insertcolor": self.colors["dark"]["text"],
                "borderwidth": 1,
                "relief": "flat"
            }},
            "Vertical.TScrollbar": {"configure": {
                "background": self.colors["dark"]["button_bg"],
                "troughcolor": self.colors["dark"]["primary_bg"],
                "arrowcolor": self.colors["dark"]["button_fg"]
            }},
            "Status.TLabel": {"configure": {
                "background": self.colors["dark"]["status_bg"],
                "foreground": self.colors["dark"]["status_fg"],
                "font": ("Helvetica", 9)
            }}
        })

        self.style.theme_use("light")
        self.root.configure(bg=self.colors["light"]["primary_bg"])

    def create_widgets(self):
        """Cria os elementos da interface gráfica"""
        # Frame principal com layout responsivo
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ttk.Frame(self.main_frame, style="TFrame", width=250)
        self.sidebar_frame.grid(row=0, column=0, sticky="ns", padx=(0, 1))
        self.sidebar_frame.configure(relief="flat")

        # Botão de toggle da sidebar
        self.toggle_sidebar_button = ttk.Button(
            self.sidebar_frame,
            text="☰",
            command=self.toggle_sidebar,
            style="TButton",
            width=3
        )
        self.toggle_sidebar_button.pack(pady=10, padx=10, anchor="nw")

        # Título da sidebar
        ttk.Label(
            self.sidebar_frame,
            text="GPTEco Menu",
            font=("Helvetica", 14, "bold"),
            foreground=self.colors[self.theme]["accent"]
        ).pack(pady=10)

        # Botões da sidebar
        sidebar_buttons = [
            ("📂 Carregar Conversa", self.load_conversation_from_file),
            ("❓ Ajuda", self.show_help),
            ("🔒 LGPD", self.show_lgpd),
            ("🌙 Tema", self.toggle_theme),
            ("🎓 Ensinar", self.open_teach_window),
            ("🧠 NLP Info", self.show_nlp_info)
        ]

        for text, command in sidebar_buttons:
            ttk.Button(
                self.sidebar_frame,
                text=text,
                command=command,
                style="TButton",
                width=20
            ).pack(pady=5, padx=10, fill=tk.X)

        # Frame de conteúdo principal
        self.content_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)

        # Cabeçalho
        self.header_frame = ttk.Frame(self.content_frame, style="TFrame")
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(10, 5))

        self.title_label = ttk.Label(
            self.header_frame,
            text="GPTEco - Assistente Virtual Inteligente",
            font=("Helvetica", 16, "bold"),
            foreground=self.colors[self.theme]["accent"]
        )
        self.title_label.pack(side=tk.LEFT)

        self.version_label = ttk.Label(
            self.header_frame,
            text=f"v{self.chatbot.version}",
            font=("Helvetica", 10),
            foreground=self.colors[self.theme]["text"]
        )
        self.version_label.pack(side=tk.LEFT, padx=5)

        # Área de chat
        self.chat_frame = ttk.Frame(self.content_frame, style="TFrame")
        self.chat_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        self.chat_area = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg=self.colors[self.theme]["secondary_bg"],
            fg=self.colors[self.theme]["text"],
            bd=0,
            relief="flat",
            padx=15,
            pady=15
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        self.chat_area.config(state='disabled')

        self.chat_area.tag_configure(
            "user_bubble",
            background=self.colors[self.theme]["user_bubble"],
            lmargin1=10,
            lmargin2=10,
            rmargin=50,
            spacing1=8,
            spacing3=8,
            wrap=tk.WORD
        )

        self.chat_area.tag_configure(
            "bot_bubble",
            background=self.colors[self.theme]["bot_bubble"],
            lmargin1=50,
            lmargin2=50,
            rmargin=10,
            spacing1=8,
            spacing3=8,
            wrap=tk.WORD
        )

        # Frame de entrada
        self.input_frame = ttk.Frame(self.content_frame, style="TFrame")
        self.input_frame.grid(row=2, column=0, sticky="ew", pady=(5, 10))

        self.user_input = ttk.Entry(
            self.input_frame,
            font=("Helvetica", 11),
            style="TEntry"
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.user_input.insert(0, "Digite sua mensagem...")
        self.user_input.bind("<FocusIn>", lambda event: self.clear_placeholder())
        self.user_input.bind("<FocusOut>", lambda event: self.restore_placeholder())
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = ttk.Button(
            self.input_frame,
            text="Enviar ➤",
            command=self.send_message,
            style="TButton"
        )
        self.send_button.pack(side=tk.RIGHT, padx=5)

        # Botões de ação
        self.action_frame = ttk.Frame(self.content_frame, style="TFrame")
        self.action_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        button_config = {"style": "TButton", "width": 15}

        self.clear_button = ttk.Button(
            self.action_frame,
            text="🧹 Limpar Chat",
            command=self.clear_chat,
            **button_config
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.save_json_button = ttk.Button(
            self.action_frame,
            text="💾 Salvar (JSON)",
            command=lambda: self.save_conversation("json"),
            **button_config
        )
        self.save_json_button.pack(side=tk.LEFT, padx=5)

        self.save_txt_button = ttk.Button(
            self.action_frame,
            text="📝 Salvar (TXT)",
            command=lambda: self.save_conversation("txt"),
            **button_config
        )
        self.save_txt_button.pack(side=tk.LEFT, padx=5)

        # Barra de status
        self.status_frame = ttk.Frame(self.content_frame, style="TFrame")
        self.status_frame.grid(row=4, column=0, sticky="ew", pady=(0, 5))

        self.status_label = ttk.Label(
            self.status_frame,
            textvariable=self.status_message,
            style="Status.TLabel",
            anchor="w",
            padding=(10, 5)
        )
        self.status_label.pack(fill=tk.X)

        # Rodapé
        self.footer_frame = ttk.Frame(self.content_frame, style="TFrame")
        self.footer_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        self.footer_label = ttk.Label(
            self.footer_frame,
            text="GPTEco Chatbot © 2025 - Assistente Virtual Inteligente | ",
            font=("Helvetica", 8),
            foreground=self.colors[self.theme]["text"]
        )
        self.footer_label.pack(side=tk.LEFT)

        self.support_link = ttk.Label(
            self.footer_frame,
            text="Suporte",
            font=("Helvetica", 8, "underline"),
            foreground=self.colors[self.theme]["accent"],
            cursor="hand2"
        )
        self.support_link.pack(side=tk.LEFT)
        self.support_link.bind("<Button-1>", lambda e: webbrowser.open("mailto:support@gpteco.com"))

    def toggle_sidebar(self):
        """Alterna a visibilidade da sidebar"""
        if self.sidebar_visible:
            self.sidebar_frame.grid_remove()
            self.toggle_sidebar_button.configure(text="☰")
        else:
            self.sidebar_frame.grid(row=0, column=0, sticky="ns", padx=(0, 1))
            self.toggle_sidebar_button.configure(text="✕")
        self.sidebar_visible = not self.sidebar_visible
        self.status_message.set("Barra lateral " + ("escondida" if not self.sidebar_visible else "exibida"))
        self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def clear_placeholder(self):
        """Remove o texto placeholder do campo de entrada"""
        if self.user_input.get() == "Digite sua mensagem...":
            self.user_input.delete(0, tk.END)

    def restore_placeholder(self):
        """Restaura o placeholder se o campo estiver vazio"""
        if not self.user_input.get():
            self.user_input.insert(0, "Digite sua mensagem...")

    def toggle_theme(self):
        """Alterna entre tema claro e escuro"""
        self.theme = "dark" if self.theme == "light" else "light"
        self.style.theme_use(self.theme)

        self.chat_area.configure(
            bg=self.colors[self.theme]["secondary_bg"],
            fg=self.colors[self.theme]["text"]
        )

        self.chat_area.tag_configure(
            "user_bubble",
            background=self.colors[self.theme]["user_bubble"]
        )

        self.chat_area.tag_configure(
            "bot_bubble",
            background=self.colors[self.theme]["bot_bubble"]
        )

        self.title_label.configure(foreground=self.colors[self.theme]["accent"])
        self.version_label.configure(foreground=self.colors[self.theme]["text"])
        self.footer_label.configure(foreground=self.colors[self.theme]["text"])
        self.support_link.configure(foreground=self.colors[self.theme]["accent"])
        self.status_label.configure(style="Status.TLabel")
        self.sidebar_frame.configure(style="TFrame")

        self.status_message.set(f"Tema alterado para {self.theme.title()}")
        logging.info(f"Tema alterado para {self.theme}")
        self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def show_help(self):
        """Exibe a janela de ajuda com informações detalhadas"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Ajuda do GPTEco Chatbot")
        help_window.geometry("800x600")
        help_window.resizable(False, False)
        self.center_child_window(help_window)

        notebook = ttk.Notebook(help_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Aba de uso básico
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Como Usar")

        basic_text = tk.Text(
            basic_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg=self.colors[self.theme]["secondary_bg"],
            fg=self.colors[self.theme]["text"],
            padx=10,
            pady=10,
            bd=0
        )
        basic_text.pack(fill=tk.BOTH, expand=True)

        basic_text.insert(tk.END, "📌 Como Usar o GPTEco Chatbot\n\n", "header")
        basic_text.insert(tk.END,
                          "O GPTEco é um assistente virtual avançado com capacidades de Processamento de Linguagem Natural (NLP).\n\n")

        basic_text.insert(tk.END, "🔹 Primeiros Passos:\n", "subheader")
        basic_text.insert(tk.END, "• Digite sua pergunta no campo abaixo e pressione Enter ou clique em 'Enviar'.\n")
        basic_text.insert(tk.END, "• Use a barra lateral para carregar conversas salvas ou acessar configurações.\n")
        basic_text.insert(tk.END, "• Ensine novas respostas com o botão 'Ensinar'.\n")
        basic_text.insert(tk.END, "• Explore as capacidades de NLP com Ctrl+N.\n\n")

        basic_text.insert(tk.END, "🔹 Exemplos de Perguntas:\n", "subheader")
        basic_text.insert(tk.END, "• Qual é a capital da França?\n")
        basic_text.insert(tk.END, "• Estou feliz, conte uma piada!\n")
        basic_text.insert(tk.END, "• O que Oscar Niemeyer fez em Brasília?\n")
        basic_text.insert(tk.END, "• Como aprender programação?\n\n")

        basic_text.insert(tk.END, "🔹 Atalhos do Teclado:\n", "subheader")
        basic_text.insert(tk.END, "• Ctrl+H: Abrir esta ajuda\n")
        basic_text.insert(tk.END, "• Ctrl+L: Limpar o chat\n")
        basic_text.insert(tk.END, "• Ctrl+T: Alternar tema\n")
        basic_text.insert(tk.END, "• Ctrl+S: Salvar conversa (JSON)\n")
        basic_text.insert(tk.END, "• Ctrl+O: Carregar conversa\n")
        basic_text.insert(tk.END, "• Ctrl+P: Ver LGPD\n")
        basic_text.insert(tk.END, "• Ctrl+N: Informações sobre NLP\n")

        basic_text.tag_configure("header", font=("Helvetica", 14, "bold"))
        basic_text.tag_configure("subheader", font=("Helvetica", 12, "bold"))
        basic_text.config(state=tk.DISABLED)

        # Aba de funcionalidades avançadas
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Funcionalidades")

        advanced_text = tk.Text(
            advanced_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg=self.colors[self.theme]["secondary_bg"],
            fg=self.colors[self.theme]["text"],
            padx=10,
            pady=10,
            bd=0
        )
        advanced_text.pack(fill=tk.BOTH, expand=True)

        advanced_text.insert(tk.END, "⚙️ Funcionalidades Avançadas\n\n", "header")
        advanced_text.insert(tk.END, "🔹 Ensino:\n", "subheader")
        advanced_text.insert(tk.END, "• Adicione novas perguntas e respostas via 'Ensinar'.\n")
        advanced_text.insert(tk.END, "• Use '|' para múltiplos padrões (ex.: oi|olá).\n\n")

        advanced_text.insert(tk.END, "🔹 Carregar Conversas:\n", "subheader")
        advanced_text.insert(tk.END, "• Use 'Carregar Conversa' para importar conversas em JSON.\n")
        advanced_text.insert(tk.END, "• As mensagens serão exibidas no chat.\n\n")

        advanced_text.insert(tk.END, "🔹 Salvamento:\n", "subheader")
        advanced_text.insert(tk.END, "• Salve conversas em JSON ou TXT.\n")
        advanced_text.insert(tk.END, "• Salvamento automático a cada 5 mensagens.\n\n")

        advanced_text.insert(tk.END, "🔹 Pesquisa na Web:\n", "subheader")
        advanced_text.insert(tk.END, "• Se eu não souber a resposta, pesquiso na web usando a API do DuckDuckGo.\n")
        advanced_text.insert(tk.END, "• Exemplo: 'Qual é o clima em São Paulo?'\n\n")

        advanced_text.insert(tk.END, "🔹 Processamento de Linguagem Natural:\n", "subheader")
        advanced_text.insert(tk.END, "• Análise de sentimento: Detecta emoções nas mensagens.\n")
        advanced_text.insert(tk.END, "• Extração de palavras-chave: Identifica termos importantes.\n")
        advanced_text.insert(tk.END, "• Reconhecimento de entidades: Detecta locais, pessoas e datas.\n")
        advanced_text.insert(tk.END, "• Contexto: Mantém o contexto da conversa para respostas mais naturais.\n")

        advanced_text.tag_configure("header", font=("Helvetica", 14, "bold"))
        advanced_text.tag_configure("subheader", font=("Helvetica", 12, "bold"))
        advanced_text.config(state=tk.DISABLED)

        self.status_message.set("Janela de ajuda aberta")
        self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def show_lgpd(self):
        """Exibe a janela de LGPD com informações detalhadas"""
        lgpd_window = tk.Toplevel(self.root)
        lgpd_window.title("LGPD - GPTEco Chatbot")
        lgpd_window.geometry("800x600")
        lgpd_window.resizable(False, False)
        self.center_child_window(lgpd_window)

        notebook = ttk.Notebook(lgpd_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Aba de política de privacidade
        privacy_frame = ttk.Frame(notebook)
        notebook.add(privacy_frame, text="Política de Privacidade")

        privacy_text = tk.Text(
            privacy_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg=self.colors[self.theme]["secondary_bg"],
            fg=self.colors[self.theme]["text"],
            padx=10,
            pady=10,
            bd=0
        )
        privacy_text.pack(fill=tk.BOTH, expand=True)

        privacy_text.insert(tk.END, "🔒 Conformidade com a LGPD\n\n", "header")
        privacy_text.insert(tk.END,
                            "O GPTEco está em total conformidade com a Lei Geral de Proteção de Dados (LGPD - Lei nº 13.709/2018). Veja como protegemos seus dados:\n\n")

        privacy_text.insert(tk.END, "🔹 Coleta de Dados:\n", "subheader")
        privacy_text.insert(tk.END,
                            "• Coletamos apenas mensagens digitadas, respostas do bot e contexto de conversa, armazenados localmente.\n")
        privacy_text.insert(tk.END, "• A base de conhecimento é atualizada em 'knowledge_base.json'.\n")
        privacy_text.insert(tk.END, "• Não coletamos informações pessoais, como nome ou e-mail.\n\n")

        privacy_text.insert(tk.END, "🔹 Uso dos Dados:\n", "subheader")
        privacy_text.insert(tk.END,
                            "• Os dados são usados exclusivamente para fornecer respostas, melhorar o bot e processar linguagem natural.\n")
        privacy_text.insert(tk.END, "• Nenhuma informação é compartilhada com terceiros.\n\n")

        privacy_text.insert(tk.END, "🔹 Armazenamento:\n", "subheader")
        privacy_text.insert(tk.END,
                            "• Todos os dados (conversas, base de conhecimento, contexto) são armazenados no seu dispositivo.\n")
        privacy_text.insert(tk.END,
                            "• Você pode excluir tudo com 'Limpar Chat' ou deletando os arquivos manualmente.\n\n")

        privacy_text.insert(tk.END, "🔹 Segurança:\n", "subheader")
        privacy_text.insert(tk.END,
                            "• Processamento local, exceto para buscas na web (que não armazenam dados pessoais).\n")
        privacy_text.insert(tk.END, "• Logs locais não incluem informações sensíveis.\n")

        privacy_text.tag_configure("header", font=("Helvetica", 14, "bold"))
        privacy_text.tag_configure("subheader", font=("Helvetica", 12, "bold"))
        privacy_text.config(state=tk.DISABLED)

        # Aba de direitos do usuário
        rights_frame = ttk.Frame(notebook)
        notebook.add(rights_frame, text="Seus Direitos")

        rights_text = tk.Text(
            rights_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg=self.colors[self.theme]["secondary_bg"],
            fg=self.colors[self.theme]["text"],
            padx=10,
            pady=10,
            bd=0
        )
        rights_text.pack(fill=tk.BOTH, expand=True)

        rights_text.insert(tk.END, "🛡️ Seus Direitos sob a LGPD\n\n", "header")
        rights_text.insert(tk.END, "Você tem os seguintes direitos garantidos pela LGPD:\n\n")

        rights_text.insert(tk.END, "🔹 Acesso:\n", "subheader")
        rights_text.insert(tk.END, "• Visualize o histórico de conversa em JSON/TXT ou na interface.\n\n")

        rights_text.insert(tk.END, "🔹 Exclusão:\n", "subheader")
        rights_text.insert(tk.END, "• Apague o histórico com 'Limpar Chat'.\n")
        rights_text.insert(tk.END, "• Delete arquivos JSON/TXT manualmente.\n\n")

        rights_text.insert(tk.END, "🔹 Consentimento:\n", "subheader")
        rights_text.insert(tk.END, "• Você controla o salvamento de conversas e a base de conhecimento.\n")
        rights_text.insert(tk.END, "• Buscas na web e processamento NLP não armazenam dados pessoais.\n\n")

        rights_text.insert(tk.END, "📧 Contato:\n", "subheader")
        rights_text.insert(tk.END, "Dúvidas? Envie um e-mail para ")
        rights_text.insert(tk.END, "privacy@gpteco.com", "link")
        rights_text.insert(tk.END, ".\n")

        rights_text.tag_configure("header", font=("Helvetica", 14, "bold"))
        rights_text.tag_configure("subheader", font=("Helvetica", 12, "bold"))
        rights_text.tag_configure("link", foreground=self.colors[self.theme]["accent"], underline=True)
        rights_text.tag_bind("link", "<Button-1>", lambda e: webbrowser.open("mailto:privacy@gpteco.com"))
        rights_text.config(state=tk.DISABLED)

        self.status_message.set("Janela de LGPD aberta")
        self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def show_nlp_info(self):
        """Exibe informações sobre as capacidades de NLP"""
        nlp_window = tk.Toplevel(self.root)
        nlp_window.title("Processamento de Linguagem Natural - GPTEco")
        nlp_window.geometry("800x600")
        nlp_window.resizable(False, False)
        self.center_child_window(nlp_window)

        notebook = ttk.Notebook(nlp_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Aba de visão geral do NLP
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Visão Geral")

        overview_text = tk.Text(
            overview_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg=self.colors[self.theme]["secondary_bg"],
            fg=self.colors[self.theme]["text"],
            padx=10,
            pady=10,
            bd=0
        )
        overview_text.pack(fill=tk.BOTH, expand=True)

        overview_text.insert(tk.END, "🧠 Processamento de Linguagem Natural (NLP)\n\n", "header")
        overview_text.insert(tk.END,
                             "O GPTEco utiliza técnicas de NLP para entender e responder de forma mais natural às suas mensagens.\n\n")

        overview_text.insert(tk.END, "🔹 O que é NLP?\n", "subheader")
        overview_text.insert(tk.END,
                             "• NLP é o campo da IA que permite às máquinas entender, interpretar e responder em linguagem humana.\n")
        overview_text.insert(tk.END, "• No GPTEco, usamos NLP para melhorar a interação com o usuário.\n\n")

        overview_text.insert(tk.END, "🔹 Funcionalidades de NLP:\n", "subheader")
        overview_text.insert(tk.END, "• Análise de Sentimento: Detecta emoções (positivo, neutro, negativo).\n")
        overview_text.insert(tk.END, "• Extração de Palavras-chave: Identifica termos importantes na mensagem.\n")
        overview_text.insert(tk.END, "• Reconhecimento de Entidades: Detecta locais, pessoas e datas.\n")
        overview_text.insert(tk.END, "• Contexto: Mantém o contexto da conversa para respostas mais coerentes.\n\n")

        overview_text.insert(tk.END, "🔹 Exemplo:\n", "subheader")
        overview_text.insert(tk.END,
                             "• Pergunta: 'Estou feliz, conte uma piada sobre Brasília!'\n")
        overview_text.insert(tk.END,
                             "• Resposta: Detecta sentimento positivo, reconhece 'Brasília' como local e adapta a resposta.\n")

        overview_text.tag_configure("header", font=("Helvetica", 14, "bold"))
        overview_text.tag_configure("subheader", font=("Helvetica", 12, "bold"))
        overview_text.config(state=tk.DISABLED)

        # Aba de demonstração de NLP
        demo_frame = ttk.Frame(notebook)
        notebook.add(demo_frame, text="Demonstração")

        demo_text = tk.Text(
            demo_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg=self.colors[self.theme]["secondary_bg"],
            fg=self.colors[self.theme]["text"],
            padx=10,
            pady=10,
            bd=0
        )
        demo_text.pack(fill=tk.BOTH, expand=True)

        demo_text.insert(tk.END, "🚀 Demonstração de NLP\n\n", "header")
        demo_text.insert(tk.END, "🔹 Como Testar:\n", "subheader")
        demo_text.insert(tk.END, "• Tente mensagens com emoções claras, como 'Estou triste' ou 'Que dia ótimo!'.\n")
        demo_text.insert(tk.END, "• Inclua locais, pessoas ou datas, como 'O que Oscar Niemeyer fez em 1960?'.\n")
        demo_text.insert(tk.END, "• Faça perguntas em sequência sobre o mesmo tópico para testar o contexto.\n\n")

        demo_text.insert(tk.END, "🔹 Exemplos Práticos:\n", "subheader")
        demo_text.insert(tk.END, "• 'Estou feliz, qual é a capital do Brasil?' → Resposta com tom positivo.\n")
        demo_text.insert(tk.END, "• 'Fale sobre Tóquio' → Reconhece 'Tóquio' como local.\n")
        demo_text.insert(tk.END, "• 'Conte mais sobre IA' (após perguntar sobre Machine Learning) → Usa contexto.\n\n")

        demo_text.insert(tk.END, "🔹 Dica:\n", "subheader")
        demo_text.insert(tk.END,
                         "• O GPTEco combina regex com similaridade de palavras-chave para respostas mais precisas.\n")
        demo_text.insert(tk.END,
                         "• Use frases naturais para ver o NLP em ação!\n")

        demo_text.tag_configure("header", font=("Helvetica", 14, "bold"))
        demo_text.tag_configure("subheader", font=("Helvetica", 12, "bold"))
        demo_text.config(state=tk.DISABLED)

        self.status_message.set("Janela de informações de NLP aberta")
        self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def center_child_window(self, child_window):
        """Centraliza uma janela filha em relação à janela principal"""
        self.root.update_idletasks()
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        child_window.update_idletasks()
        c_width = child_window.winfo_width()
        c_height = child_window.winfo_height()

        pos_x = x + (width // 2) - (c_width // 2)
        pos_y = y + (height // 2) - (c_height // 2)

        child_window.geometry(f"+{pos_x}+{pos_y}")

    def open_teach_window(self):
        """Abre uma janela para ensinar novas perguntas/respostas"""
        teach_window = tk.Toplevel(self.root)
        teach_window.title("Ensinar o GPTEco")
        teach_window.geometry("600x500")
        teach_window.resizable(False, False)
        teach_window.configure(bg=self.colors[self.theme]["primary_bg"])
        self.center_child_window(teach_window)

        ttk.Label(
            teach_window,
            text="Ensine uma nova resposta ao GPTEco",
            font=("Helvetica", 14, "bold"),
            foreground=self.colors[self.theme]["accent"]
        ).pack(pady=10)

        ttk.Label(
            teach_window,
            text="Digite as palavras-chave (padrão) e a resposta correspondente.",
            font=("Helvetica", 10),
            foreground=self.colors[self.theme]["text"]
        ).pack(pady=5)

        ttk.Label(
            teach_window,
            text="Padrão (ex.: oi|olá|bom dia):",
            font=("Helvetica", 11),
            foreground=self.colors[self.theme]["text"]
        ).pack(pady=5)

        pattern_entry = ttk.Entry(
            teach_window,
            font=("Helvetica", 11),
            width=50
        )
        pattern_entry.pack(pady=5)

        ttk.Label(
            teach_window,
            text="Dica: Use '|' para múltiplas opções ou palavras-chave simples.",
            font=("Helvetica", 9, "italic"),
            foreground=self.colors[self.theme]["text"]
        ).pack(pady=5)

        ttk.Label(
            teach_window,
            text="Resposta:",
            font=("Helvetica", 11),
            foreground=self.colors[self.theme]["text"]
        ).pack(pady=5)

        response_text = tk.Text(
            teach_window,
            height=5,
            width=50,
            font=("Helvetica", 11),
            bg=self.colors[self.theme]["input_bg"],
            fg=self.colors[self.theme]["text"],
            insertbackground=self.colors[self.theme]["text"]
        )
        response_text.pack(pady=5)

        def submit_teach():
            pattern = pattern_entry.get().strip()
            response = response_text.get("1.0", tk.END).strip()

            if not pattern or not response:
                messagebox.showwarning(
                    "Aviso",
                    "Padrão e resposta não podem estar vazios!",
                    parent=teach_window
                )
                return

            if self.chatbot.add_knowledge(pattern, response):
                messagebox.showinfo(
                    "Sucesso",
                    "Novo conhecimento adicionado! Testando a pergunta agora...",
                    parent=teach_window
                )
                teach_window.destroy()
                self.send_message_test(pattern)
                self.status_message.set("Novo conhecimento adicionado")
                self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))
            else:
                messagebox.showerror(
                    "Erro",
                    "Falha ao adicionar conhecimento. Verifique o padrão regex.",
                    parent=teach_window
                )
                self.status_message.set("Erro ao adicionar conhecimento")
                self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

        button_frame = ttk.Frame(teach_window)
        button_frame.pack(pady=20)

        ttk.Button(
            button_frame,
            text="Adicionar",
            command=submit_teach,
            style="TButton"
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancelar",
            command=teach_window.destroy,
            style="TButton"
        ).pack(side=tk.LEFT, padx=5)

        self.status_message.set("Janela de ensino aberta")
        self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def send_message_test(self, test_message):
        """Envia uma mensagem de teste após ensinar"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chatbot.conversation_history.append({
            "user": test_message,
            "time": timestamp
        })
        self.display_message(f"👤 Você ({timestamp}): {test_message}", "user")
        self.process_message(test_message, timestamp)

    def send_message(self, event=None):
        """Processa a mensagem do usuário e exibe a resposta"""
        user_message = self.user_input.get().strip()
        if not user_message or user_message == "Digite sua mensagem...":
            messagebox.showwarning(
                "Aviso",
                "Por favor, digite uma mensagem válida!",
                parent=self.root
            )
            self.status_message.set("Mensagem inválida")
            self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chatbot.conversation_history.append({
            "user": user_message,
            "time": timestamp
        })

        self.display_message(f"👤 Você ({timestamp}): {user_message}", "user")
        self.user_input.delete(0, tk.END)
        self.status_message.set("Processando mensagem com NLP...")
        self.animate_typing(timestamp)

    def animate_typing(self, timestamp):
        """Simula a animação de 'digitando...' com pontos piscantes"""
        typing_msg = f"🤖 {self.chatbot.bot_name}: Processando"
        self.display_message(typing_msg, "bot")

        def update_dots(count=0):
            dots = "." * (count % 4)
            self.chat_area.config(state='normal')
            if "Processando" in self.chat_area.get("end-3l", "end-1l"):
                self.chat_area.delete("end-3l", "end-1l")
                self.display_message(f"🤖 {self.chatbot.bot_name}: Processando{dots}", "bot")
            self.chat_area.config(state='disabled')
            if count < 4:
                self.root.after(300, update_dots, count + 1)
            else:
                threading.Thread(
                    target=self.process_message,
                    args=(self.chatbot.conversation_history[-1]["user"], timestamp),
                    daemon=True
                ).start()

        update_dots()

    def process_message(self, user_message, timestamp):
        """Processa a mensagem em uma thread separada"""
        time.sleep(0.3)
        response = self.chatbot.find_response(user_message)

        if not response:
            self.status_message.set("Pesquisando na web...")
            response = self.chatbot.search_web(user_message)

        self.chatbot.conversation_history.append({
            "bot": response,
            "time": timestamp
        })

        self.root.after(0, self.update_chat_area, f"🤖 {self.chatbot.bot_name} ({timestamp}): {response}", "bot")

        if len(self.chatbot.conversation_history) % 5 == 0:
            self.save_conversation("json", silent=True)

        self.status_message.set("Resposta enviada")
        self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def update_chat_area(self, message, sender):
        """Atualiza a área de chat com uma nova mensagem"""
        self.chat_area.config(state='normal')
        if "Processando" in self.chat_area.get("end-3l", "end-1l"):
            self.chat_area.delete("end-3l", "end-1l")

        tag = "user_bubble" if sender == "user" else "bot_bubble"
        self.chat_area.insert(tk.END, f"{message}\n\n", tag)
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)

    def display_message(self, message, sender):
        """Exibe uma mensagem na área de chat"""
        self.chat_area.config(state='normal')
        tag = "user_bubble" if sender == "user" else "bot_bubble"
        self.chat_area.insert(tk.END, f"{message}\n\n", tag)
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)

    def clear_chat(self):
        """Limpa a área de chat, histórico e contexto"""
        if messagebox.askyesno(
                "Confirmar",
                "Tem certeza que deseja limpar o chat? Isso apagará o histórico e o contexto da conversa.",
                parent=self.root
        ):
            self.chat_area.config(state='normal')
            self.chat_area.delete(1.0, tk.END)
            self.chat_area.config(state='disabled')
            self.chatbot.conversation_history = []
            self.chatbot.context.clear()
            self.display_welcome_message()
            logging.info("Chat e contexto limpos pelo usuário")
            self.status_message.set("Chat e contexto limpos")
            self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def save_conversation(self, format="json", silent=False):
        """Salva a conversa em JSON ou TXT"""
        success, result = self.chatbot.save_conversation(format)

        if success:
            if not silent:
                messagebox.showinfo(
                    "Sucesso",
                    f"Conversa salva em:\n{result}",
                    parent=self.root
                )
                self.status_message.set(f"Conversa salva em {result}")
            logging.info(f"Conversa salva em {result}")
        else:
            messagebox.showerror(
                "Erro",
                f"Falha ao salvar a conversa:\n{result}",
                parent=self.root
            )
            self.status_message.set("Erro ao salvar conversa")
            logging.error(f"Falha ao salvar conversa: {result}")

        if not silent:
            self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def load_conversation_from_file(self):
        """Abre um seletor de arquivos para carregar uma conversa em JSON"""
        filename = filedialog.askopenfilename(
            title="Selecionar Conversa",
            filetypes=[("Arquivos JSON", "*.json")],
            parent=self.root
        )

        if not filename:
            return

        conversation = self.chatbot.load_conversation(filename)
        if conversation is None:
            messagebox.showerror(
                "Erro",
                "Não foi possível carregar a conversa. Verifique o arquivo.",
                parent=self.root
            )
            self.status_message.set("Erro ao carregar conversa")
            self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))
            return

        self.chat_area.config(state='normal')
        self.chat_area.delete(1.0, tk.END)
        self.chatbot.conversation_history = conversation

        # Reconstruir contexto a partir da conversa carregada
        for entry in conversation:
            if "user" in entry:
                keywords = self.chatbot.extract_keywords(entry["user"])
                entities = self.chatbot.extract_entities(entry["user"])
                self.chatbot.update_context(entry["user"], keywords, entities)
                timestamp = entry.get("time", datetime.now().strftime("%H:%M:%S"))
                self.display_message(f"👤 Você ({timestamp}): {entry['user']}", "user")
            elif "bot" in entry:
                timestamp = entry.get("time", datetime.now().strftime("%H:%M:%S"))
                self.display_message(f"🤖 {self.chatbot.bot_name} ({timestamp}): {entry['bot']}", "bot")

        self.chat_area.config(state='disabled')
        messagebox.showinfo(
            "Sucesso",
            f"Conversa carregada de:\n{filename}",
            parent=self.root
        )
        self.status_message.set(f"Conversa carregada de {filename}")
        logging.info(f"Conversa carregada de {filename}")
        self.root.after(3000, lambda: self.status_message.set("Digite sua mensagem..."))

    def on_closing(self):
        """Executa ações antes de fechar o aplicativo"""
        if self.chatbot.conversation_history and messagebox.askyesno(
                "Sair",
                "Deseja salvar a conversa antes de sair?",
                parent=self.root
        ):
            self.save_conversation("json")
        self.chatbot.save_knowledge_base()
        logging.info("Aplicativo encerrado pelo usuário")
        self.status_message.set("Encerrando aplicativo...")
        self.root.after(1000, self.root.destroy)


def main():
    try:
        root = tk.Tk()
        app = GPTEcoGUI(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Erro na inicialização do aplicativo: {str(e)}")
        messagebox.showerror("Erro Fatal", "O aplicativo encontrou um erro e será encerrado.")
        raise


if __name__ == "__main__":
    main()
