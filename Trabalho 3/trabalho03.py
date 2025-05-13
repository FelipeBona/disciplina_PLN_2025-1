import os
import re
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import ttk, messagebox, scrolledtext, font
from urllib.parse import urlparse
import pyperclip
import requests
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BibleScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Extração de Textos Bíblicos")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        self.root.minsize(800, 600)

        # Configuração de estilo
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.primary_color = "#01627E"
        self.secondary_color = "#f0f0f0"
        self.text_color = "#333333"
        self.success_color = "#4CAF50"
        self.error_color = "#F44336"

        self.style.configure('TFrame', background=self.secondary_color)
        self.style.configure('TLabel', background=self.secondary_color, font=('Segoe UI', 9),
                             foreground=self.text_color)
        self.style.configure('TButton', font=('Segoe UI', 9), padding=6, background=self.primary_color,
                             foreground='white')
        self.style.map('TButton', background=[('active', '#0288D1'), ('pressed', '#005B7F')])
        self.style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'), foreground=self.primary_color)
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 9, 'italic'), foreground="#666666")
        self.style.configure('Success.TLabel', foreground=self.success_color)
        self.style.configure('Error.TLabel', foreground=self.error_color)
        self.style.configure('TEntry', padding=4, font=('Segoe UI', 9))
        self.style.configure("green.Horizontal.TProgressbar", troughcolor=self.secondary_color,
                             background=self.success_color)

        self.root.configure(bg=self.secondary_color)

        # Cabeçalho
        self.header_frame = ttk.Frame(self.root)
        self.header_frame.pack(pady=(5, 3), padx=5, fill='x')
        self.title_label = ttk.Label(self.header_frame, text="Extração de Textos Bíblicos", style='Header.TLabel')
        self.title_label.pack()
        self.subtitle_label = ttk.Label(self.header_frame, text="Ferramenta para extração e análise de textos bíblicos",
                                        style='Subtitle.TLabel')
        self.subtitle_label.pack()

        # Frame principal
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(pady=3, padx=5, fill='both', expand=True)

        # Frame de entrada
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(fill='x', pady=(3, 5))
        ttk.Label(self.input_frame, text="URL do texto bíblico (ex: https://www.bibliaonline.com.br/nvi/gn/1):").pack(
            anchor='w')
        self.url_frame = ttk.Frame(self.input_frame)
        self.url_frame.pack(fill='x', expand=True)
        self.url_entry = ttk.Entry(self.url_frame)
        self.url_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.fetch_button = ttk.Button(self.url_frame, text="Extrair", command=self.fetch_bible_text, style='TButton',
                                       width=10)
        self.fetch_button.pack(side='left')

        # Frame de status
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill='x', pady=(0, 5))
        self.status_label = ttk.Label(self.status_frame, text="Pronto para começar.")
        self.status_label.pack(anchor='w')

        # Frame de resultados
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.pack(fill='both', expand=True)
        self.result_notebook = ttk.Notebook(self.result_frame)
        self.result_notebook.pack(fill='both', expand=True)

        # Aba de texto completo
        self.text_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.text_tab, text='Texto Completo')
        ttk.Label(self.text_tab, text="Texto Extraído:").pack(anchor='w')
        text_font = font.Font(family='Segoe UI', size=10)
        self.result_text = scrolledtext.ScrolledText(self.text_tab, wrap=tk.WORD, font=text_font, padx=8, pady=8,
                                                     bg='white', fg=self.text_color, insertbackground=self.text_color,
                                                     selectbackground=self.primary_color, selectforeground='white',
                                                     state='disabled')
        self.result_text.pack(fill='both', expand=True)
        self.result_text.tag_configure('header', font=('Segoe UI', 11, 'bold'))
        self.result_text.tag_configure('verse', font=('Segoe UI', 10))
        self.result_text.tag_configure('verse_number', font=('Segoe UI', 10, 'bold'), foreground=self.primary_color)

        # Aba de palavras-chave
        self.keywords_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.keywords_tab, text='Palavras Importantes')
        ttk.Label(self.keywords_tab, text="Palavras mais relevantes (TF-IDF):").pack(anchor='w')
        self.keywords_text = scrolledtext.ScrolledText(self.keywords_tab, wrap=tk.WORD, font=text_font, padx=8, pady=8,
                                                       bg='white', fg=self.text_color, insertbackground=self.text_color,
                                                       selectbackground=self.primary_color, selectforeground='white',
                                                       state='disabled')
        self.keywords_text.pack(fill='both', expand=True)

        # Aba de clusterização
        self.cluster_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.cluster_tab, text='Análise de Tópicos')
        ttk.Label(self.cluster_tab, text="Agrupamento de versículos por similaridade:").pack(anchor='w')
        self.cluster_text = scrolledtext.ScrolledText(self.cluster_tab, wrap=tk.WORD, font=text_font, padx=8, pady=8,
                                                      bg='white', fg=self.text_color, insertbackground=self.text_color,
                                                      selectbackground=self.primary_color, selectforeground='white',
                                                      state='disabled')
        self.cluster_text.pack(fill='both', expand=True)

        # Aba de entidades
        self.entities_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.entities_tab, text='Análise de Entidades')
        ttk.Label(self.entities_tab, text="Entidades e Atributos Linguísticos:").pack(anchor='w')
        self.entities_text = scrolledtext.ScrolledText(self.entities_tab, wrap=tk.WORD, font=text_font, padx=8, pady=8,
                                                       bg='white', fg=self.text_color, insertbackground=self.text_color,
                                                       selectbackground=self.primary_color, selectforeground='white',
                                                       state='disabled')
        self.entities_text.pack(fill='both', expand=True)

        # Aba de sumarização
        self.summary_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.summary_tab, text='Sumarização')
        ttk.Label(self.summary_tab, text="Sumarização Extrativa e Palavras-chave:").pack(anchor='w')
        self.summary_text = scrolledtext.ScrolledText(self.summary_tab, wrap=tk.WORD, font=text_font, padx=8, pady=8,
                                                      bg='white', fg=self.text_color, insertbackground=self.text_color,
                                                      selectbackground=self.primary_color, selectforeground='white',
                                                      state='disabled')
        self.summary_text.pack(fill='both', expand=True)

        # Aba de classificação
        self.classification_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.classification_tab, text='Classificação')
        ttk.Label(self.classification_tab, text="Classificação de Sentimento:").pack(anchor='w')
        self.classification_text = scrolledtext.ScrolledText(self.classification_tab, wrap=tk.WORD, font=text_font,
                                                             padx=8, pady=8, bg='white', fg=self.text_color,
                                                             insertbackground=self.text_color,
                                                             selectbackground=self.primary_color,
                                                             selectforeground='white', state='disabled')
        self.classification_text.pack(fill='both', expand=True)

        # Aba de tema do capítulo
        self.theme_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.theme_tab, text='Tema do Capítulo')
        ttk.Label(self.theme_tab, text="Tema Principal do Capítulo:").pack(anchor='w')
        self.theme_text = scrolledtext.ScrolledText(self.theme_tab, wrap=tk.WORD, font=text_font, padx=8, pady=8,
                                                    bg='white', fg=self.text_color, insertbackground=self.text_color,
                                                    selectbackground=self.primary_color, selectforeground='white',
                                                    state='disabled')
        self.theme_text.pack(fill='both', expand=True)

        # Aba de adivinhação de livro
        self.guess_book_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.guess_book_tab, text='Adivinhar Livro')
        ttk.Label(self.guess_book_tab, text="Adivinhar Livro de Versículos:").pack(anchor='w')
        self.guess_book_text = scrolledtext.ScrolledText(self.guess_book_tab, wrap=tk.WORD, font=text_font, padx=8,
                                                         pady=8, bg='white', fg=self.text_color,
                                                         insertbackground=self.text_color,
                                                         selectbackground=self.primary_color, selectforeground='white',
                                                         state='disabled')
        self.guess_book_text.pack(fill='both', expand=True)

        # Frame de rodapé
        self.footer_frame = ttk.Frame(self.main_frame)
        self.footer_frame.pack(fill='x', pady=(5, 0))
        self.save_button = ttk.Button(self.footer_frame, text="Salvar", command=self.save_to_file, style='TButton',
                                      state='disabled', width=10)
        self.save_button.pack(side='left', padx=(0, 5))
        self.copy_button = ttk.Button(self.footer_frame, text="Copiar", command=self.copy_to_clipboard, style='TButton',
                                      state='disabled', width=15)
        self.copy_button.pack(side='left', padx=(0, 5))
        self.clear_button = ttk.Button(self.footer_frame, text="Limpar", command=self.clear_results, style='TButton',
                                       width=10)
        self.clear_button.pack(side='left')
        self.progress = ttk.Progressbar(self.footer_frame, orient='horizontal', mode='determinate', length=150,
                                        style="green.Horizontal.TProgressbar")
        self.progress.pack(side='right')

        # Menu
        self.menu_bar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0, bg=self.secondary_color, fg=self.text_color)
        self.file_menu.add_command(label="Salvar", command=self.save_to_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Sair", command=self.root.quit)
        self.menu_bar.add_cascade(label="Arquivo", menu=self.file_menu)
        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0, bg=self.secondary_color, fg=self.text_color)
        self.edit_menu.add_command(label="Copiar", command=self.copy_to_clipboard)
        self.edit_menu.add_command(label="Limpar", command=self.clear_results)
        self.menu_bar.add_cascade(label="Editar", menu=self.edit_menu)
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0, bg=self.secondary_color, fg=self.text_color)
        self.help_menu.add_command(label="Sobre", command=self.show_about)
        self.help_menu.add_command(label="Documentação", command=self.open_docs)
        self.menu_bar.add_cascade(label="Ajuda", menu=self.help_menu)
        self.root.config(menu=self.menu_bar)

        self.url_entry.insert(0, "https://www.bibliaonline.com.br/nvi/gn/1")
        self.root.bind('<Control-s>', lambda e: self.save_to_file())
        self.root.bind('<Control-c>', lambda e: self.copy_to_clipboard())

    def fetch_bible_text(self):
        url = self.url_entry.get().strip()
        logging.info(f"Iniciando extração para URL: {url}")

        if not url:
            self.show_error("Por favor, insira uma URL válida.")
            return

        try:
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError("URL inválida")
            logging.info(f"URL válida: {parsed_url}")

            if not re.search(r'/.+/.+/.+', parsed_url.path):
                raise ValueError("URL não segue o formato esperado: /traducao/livro/capitulo")

            self.update_status(f"Conectando a {parsed_url.netloc}...", "black")
            self.progress['value'] = 20
            self.root.update()

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7'
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            logging.info("Página baixada com sucesso.")

            self.update_status("Processando conteúdo...", "black")
            self.progress['value'] = 50
            self.root.update()

            soup = BeautifulSoup(response.text, 'html.parser')
            path_parts = [p for p in parsed_url.path.split('/') if p]
            translation = path_parts[-3] if len(path_parts) >= 3 else "desconhecida"
            book = path_parts[-2] if len(path_parts) >= 2 else "desconhecido"
            chapter = path_parts[-1] if len(path_parts) >= 1 else "desconhecido"
            logging.info(f"Metadados extraídos: tradução={translation}, livro={book}, capítulo={chapter}")

            text_content = self.extract_bible_text(soup)
            if not text_content:
                raise ValueError("Não foi possível identificar o texto bíblico na página.")
            logging.info(f"Texto bíblico extraído: {len(text_content)} versículos")

            for text_widget in [self.result_text, self.keywords_text, self.cluster_text, self.entities_text,
                                self.summary_text, self.classification_text, self.theme_text, self.guess_book_text]:
                text_widget.config(state='normal')
                text_widget.delete(1.0, tk.END)

            self.result_text.insert(tk.END, f"TRADUÇÃO: {translation.upper()}\n", 'header')
            self.result_text.insert(tk.END, f"LIVRO: {book.upper()}\n", 'header')
            self.result_text.insert(tk.END, f"CAPÍTULO: {chapter}\n\n", 'header')

            verses_text = []
            full_text = ""
            for verse in text_content:
                self.result_text.insert(tk.END, f"{verse['number']} ", 'verse_number')
                self.result_text.insert(tk.END, f"{verse['text']}\n\n", 'verse')
                verses_text.append(verse['text'])
                full_text += f" {verse['text']}"
            logging.info("Texto completo inserido na aba 'Texto Completo'")
            self.result_text.config(state='disabled')

            self.update_status("Extraindo palavras-chave...", "black")
            self.progress['value'] = 60
            self.root.update()
            keywords = self.extract_keywords(verses_text)
            self.keywords_text.insert(tk.END, "Palavras mais relevantes (TF-IDF):\n\n", 'header')
            for word, score in keywords[:50]:
                self.keywords_text.insert(tk.END, f"{word}: {score:.4f}\n")
            self.keywords_text.config(state='disabled')
            logging.info("Palavras-chave inseridas na aba 'Palavras Importantes'")

            self.update_status("Realizando análise de tópicos...", "black")
            self.progress['value'] = 70
            self.root.update()
            if len(verses_text) >= 2:
                clusters = self.cluster_verses(verses_text)
                self.show_clusters(clusters, text_content)
            else:
                self.cluster_text.insert(tk.END, "Pelo menos 2 versículos são necessários para análise de tópicos.\n")
            self.cluster_text.config(state='disabled')
            logging.info("Análise de tópicos concluída")

            self.update_status("Analisando entidades e atributos...", "black")
            self.progress['value'] = 80
            self.root.update()
            self.analyze_entities(full_text, verses_text)
            logging.info("Análise de entidades concluída")

            self.update_status("Gerando resumos...", "black")
            self.progress['value'] = 90
            self.root.update()
            self.generate_summaries(full_text, len(verses_text))
            logging.info("Sumarização concluída")

            self.update_status("Classificando texto...", "black")
            self.progress['value'] = 95
            self.root.update()
            self.classify_text(verses_text)
            logging.info("Classificação concluída")

            self.update_status("Deduzindo tema do capítulo...", "black")
            self.progress['value'] = 97
            self.root.update()
            self.deduce_chapter_theme(keywords, clusters if len(verses_text) >= 2 else [], verses_text)
            logging.info("Dedução de tema concluída")

            self.update_status("Adivinhando livro...", "black")
            self.progress['value'] = 99
            self.root.update()
            self.guess_book(verses_text)
            logging.info("Adivinhação de livro concluída")

            self.update_status(f"Texto extraído e analisado com sucesso de {parsed_url.netloc}!", self.success_color)
            self.progress['value'] = 100
            self.save_button.config(state='normal')
            self.copy_button.config(state='normal')
            logging.info("Processamento concluído com sucesso")

        except requests.exceptions.RequestException as e:
            self.show_error(f"Erro ao acessar a URL: {str(e)}")
            logging.error(f"Erro na requisição: {str(e)}")
            self.progress['value'] = 0
        except ValueError as e:
            self.show_error(str(e))
            logging.error(f"Erro de validação: {str(e)}")
            self.progress['value'] = 0
        except Exception as e:
            self.show_error(f"Erro inesperado: {str(e)}")
            logging.error(f"Erro inesperado: {str(e)}")
            self.progress['value'] = 0
        finally:
            self.root.after(2000, lambda: self.progress.config(value=0))
            logging.info("Finalizando processamento")

    def extract_bible_text(self, soup):
        try:
            verses = []
            verse_elements = soup.find_all('p', class_=re.compile(r'verse|versiculo|texto'))
            if not verse_elements:
                verse_elements = soup.find_all('p')

            current_verse = {'number': '1', 'text': ''}
            for elem in verse_elements:
                text = elem.get_text(strip=True)
                if not text:
                    continue
                verse_match = re.match(r'^(\d+)\s*(.*)$', text)
                if verse_match:
                    if current_verse['text']:
                        verses.append(current_verse)
                    current_verse = {'number': verse_match.group(1), 'text': verse_match.group(2)}
                else:
                    if current_verse['text']:
                        current_verse['text'] += ' ' + text
                    else:
                        current_verse['text'] = text

            if current_verse['text']:
                verses.append(current_verse)

            if not verses:
                full_text = soup.get_text('\n', strip=True)
                verses.append({'number': '1', 'text': full_text})
                logging.warning("Nenhum versículo identificado. Usando texto completo como fallback.")

            logging.info(f"Extraídos {len(verses)} versículos")
            return verses
        except Exception as e:
            logging.error(f"Erro ao extrair texto bíblico: {str(e)}")
            raise ValueError(f"Falha na extração do texto: {str(e)}")

    def extract_keywords(self, verses_text):
        try:
            stopwords = {
                'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na',
                'nos', 'nas', 'por', 'para', 'com', 'como', 'que', 'e', 'ou', 'se', 'mas', 'porque', 'pois', 'quando',
                'onde', 'como', 'qual', 'quais', 'quem', 'não', 'sim'
            }
            vectorizer = TfidfVectorizer(
                stop_words=list(stopwords),
                max_features=100,
                min_df=1,
                max_df=0.9
            )
            X = vectorizer.fit_transform(verses_text)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = X.max(axis=0).toarray()[0]
            keyword_scores = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
            logging.info(f"Extraídas {len(keyword_scores)} palavras-chave via TF-IDF")
            return keyword_scores
        except Exception as e:
            logging.error(f"Erro ao extrair palavras-chave: {str(e)}")
            return []

    def cluster_verses(self, verses_text):
        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words=['o', 'a', 'os', 'as', 'um', 'uma', 'de', 'do', 'da', 'em', 'no', 'na', 'por', 'para', 'com',
                            'que', 'e', 'se'],
                min_df=1,
                max_df=0.9
            )
            processed_texts = [re.sub(r'\d+', '', text.lower()) for text in verses_text]
            processed_texts = [re.sub(r'[^\w\s]', '', text) for text in processed_texts]
            X = vectorizer.fit_transform(processed_texts)
            n_verses = len(verses_text)
            n_clusters = min(4, max(2, n_verses // 2))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init=10, max_iter=300)
            kmeans.fit(X)
            logging.info(f"Clusterização concluída com {n_clusters} clusters")
            return kmeans.labels_
        except Exception as e:
            logging.error(f"Erro na clusterização: {str(e)}")
            return np.zeros(len(verses_text), dtype=int)

    def show_clusters(self, clusters, verses_data):
        try:
            unique_clusters = set(clusters)
            self.cluster_text.insert(tk.END, "Agrupamento de Versículos por Tópicos\n\n", 'header')
            for cluster_id in sorted(unique_clusters):
                self.cluster_text.insert(tk.END, f"\n=== Tópico {cluster_id + 1} ===\n", 'header')
                cluster_verses = [v for v, c in zip(verses_data, clusters) if c == cluster_id]
                for verse in cluster_verses:
                    self.cluster_text.insert(tk.END, f"{verse['number']} ", 'verse_number')
                    self.cluster_text.insert(tk.END, f"{verse['text']}\n\n", 'verse')
                self.cluster_text.insert(tk.END, f"Total de versículos neste tópico: {len(cluster_verses)}\n")
            logging.info("Clusters exibidos com sucesso")
        except Exception as e:
            self.cluster_text.insert(tk.END, f"Erro ao exibir clusters: {str(e)}\n")
            logging.error(f"Erro ao exibir clusters: {str(e)}")

    def analyze_entities(self, text, verses_text):
        try:
            self.entities_text.config(state='normal')
            self.entities_text.delete(1.0, tk.END)

            # Dicionário de entidades pré-definidas
            entity_dict = {
                'PESSOA': ['Deus', 'Adão', 'Eva', 'Noé', 'Moisés', 'Jesus', 'Paulo', 'Abraão', 'Isaque', 'Jacó', 'Davi',
                           'Salomão'],
                'LOCAL': ['terra', 'céus', 'Egito', 'Israel', 'Jerusalém', 'Sinai', 'Canaã', 'Babilônia', 'Jordão'],
                'ORGANIZAÇÃO': ['Israel', 'Igreja', 'Sacerdotes', 'Levitas', 'Fariseus']
            }

            # Dicionário de categorias gramaticais
            pos_patterns = {
                'SUBSTANTIVO': r'\b(terra|céus|luz|águas|homem|povo|lei|coração|amor|vida|dia|noite|mar|sol|lua)\b',
                'VERBO': r'\b(criou|disse|fez|viu|chamou|amou|andou|julgou|ordenou|abençoou|castigou)\b',
                'ADJETIVO': r'\b(bom|santo|justo|sábio|eterno|verdadeiro|glorioso|sagrado)\b'
            }

            # Extração de entidades
            entities = {'PESSOA': [], 'LOCAL': [], 'ORGANIZAÇÃO': []}
            for ent_type, ent_list in entity_dict.items():
                for ent in ent_list:
                    if re.search(rf'\b{ent}\b', text, re.IGNORECASE):
                        entities[ent_type].append(ent)

            # Extração de atributos gramaticais
            pos_counts = {'SUBSTANTIVO': [], 'VERBO': [], 'ADJETIVO': []}
            for pos_type, pattern in pos_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                pos_counts[pos_type].extend(matches)

            # Análise estatística
            ent_counts = {k: Counter(v).most_common() for k, v in entities.items()}
            pos_counts = {k: Counter(v).most_common() for k, v in pos_counts.items()}
            total_ents = sum(len(v) for v in entities.values())
            total_pos = sum(len(v) for v in pos_counts.items())

            # Frequência por versículo
            ent_freq_per_verse = []
            for verse in verses_text:
                verse_ents = {'PESSOA': 0, 'LOCAL': 0, 'ORGANIZAÇÃO': 0}
                for ent_type, ent_list in entity_dict.items():
                    for ent in ent_list:
                        if re.search(rf'\b{ent}\b', verse, re.IGNORECASE):
                            verse_ents[ent_type] += 1
                ent_freq_per_verse.append(verse_ents)

            logging.info(f"Entidades extraídas: {total_ents}, Atributos: {total_pos}")

            # Exibir resultados
            self.entities_text.insert(tk.END, "Análise de Entidades e Atributos Linguísticos\n\n", 'header')
            self.entities_text.insert(tk.END, "=== Entidades Nomeadas ===\n")
            for ent_type, counts in ent_counts.items():
                self.entities_text.insert(tk.END, f"\n{ent_type}:\n")
                if counts:
                    for entity, count in counts:
                        self.entities_text.insert(tk.END, f"  {entity}: {count}\n")
                else:
                    self.entities_text.insert(tk.END, "  Nenhuma encontrada\n")
                self.entities_text.insert(tk.END, f"Total de {ent_type}: {len(entities[ent_type])}\n")
            self.entities_text.insert(tk.END, f"\nTotal de entidades: {total_ents}\n")

            self.entities_text.insert(tk.END, "\n=== Atributos Linguísticos ===\n")
            for pos_type, counts in pos_counts.items():
                self.entities_text.insert(tk.END, f"\n{pos_type}:\n")
                if counts:
                    for word, count in counts:
                        self.entities_text.insert(tk.END, f"  {word}: {count}\n")
                else:
                    self.entities_text.insert(tk.END, "  Nenhum encontrado\n")
                self.entities_text.insert(tk.END, f"Total de {pos_type}: {len(pos_counts[pos_type])}\n")
            self.entities_text.insert(tk.END, f"\nTotal de atributos: {total_pos}\n")

            self.entities_text.insert(tk.END, "\n=== Frequência por Versículo ===\n")
            for i, freq in enumerate(ent_freq_per_verse, 1):
                self.entities_text.insert(tk.END,
                                          f"Versículo {i}: PESSOA={freq['PESSOA']}, LOCAL={freq['LOCAL']}, ORGANIZAÇÃO={freq['ORGANIZAÇÃO']}\n")

            self.entities_text.insert(tk.END, "\n=== Importância para Textos Bíblicos ===\n")
            self.entities_text.insert(tk.END,
                                      "A identificação de entidades como 'Deus' (PESSOA), 'Israel' (LOCAL/ORGANIZAÇÃO) e 'Jerusalém' (LOCAL) é crucial para contextualizar narrativas bíblicas, revelando agentes, cenários e instituições centrais. Atributos como 'criou' (VERBO) e 'santo' (ADJETIVO) destacam ações e qualidades teológicas. A análise estatística mostra a distribuição de temas, como a predominância de 'Deus' em Gênesis, indicando foco na criação divina. A frequência por versículo ajuda a mapear a densidade de entidades, útil para estudos exegéticos.\n")
            logging.info("Resultados de entidades exibidos")
        except Exception as e:
            self.entities_text.insert(tk.END, f"Erro ao processar entidades: {str(e)}\n")
            logging.error(f"Erro ao processar entidades: {str(e)}")
            self.entities_text.insert(tk.END, "\n=== Fallback: Palavras-Chave ===\n")
            keywords = self.extract_keywords(verses_text)
            for word, score in keywords[:5]:
                self.entities_text.insert(tk.END, f"{word}: {score:.4f}\n")
        finally:
            self.entities_text.config(state='disabled')

    def generate_summaries(self, text, num_verses):
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("portuguese"))
            num_sentences = max(3, num_verses // 5)
            lex_rank = LexRankSummarizer()
            lex_summary = lex_rank(parser.document, num_sentences)
            lex_summary_text = " ".join(str(sentence) for sentence in lex_summary)
            lsa_summarizer = LsaSummarizer()
            lsa_summary = lsa_summarizer(parser.document, num_sentences)
            lsa_summary_text = " ".join(str(sentence) for sentence in lsa_summary)
            keywords = self.extract_keywords([text])
            top_keywords = [word for word, _ in keywords[:10]]
            self.summary_text.insert(tk.END, "Sumarização e Extração de Palavras-chave\n\n", 'header')
            self.summary_text.insert(tk.END, "=== Sumarização Extrativa (LexRank) ===\n")
            self.summary_text.insert(tk.END, f"{lex_summary_text}\n\n")
            self.summary_text.insert(tk.END, "=== Sumarização Extrativa (LSA) ===\n")
            self.summary_text.insert(tk.END, f"{lsa_summary_text}\n\n")
            self.summary_text.insert(tk.END, "=== Palavras-chave (TF-IDF) ===\n")
            self.summary_text.insert(tk.END, ", ".join(top_keywords) + "\n\n")
            self.summary_text.insert(tk.END, "=== Relevância para Textos Bíblicos ===\n")
            self.summary_text.insert(tk.END,
                                     "LexRank é eficaz para destacar eventos centrais, como a criação em Gênesis 1, capturando versículos narrativos. LSA identifica temas latentes, como a relação entre Deus e a criação, útil para análises teológicas. Palavras-chave via TF-IDF revelam termos centrais (ex.: 'Deus', 'terra'), permitindo comparações entre capítulos. Essas abordagens são valiosas para resumir capítulos longos e identificar ideias principais, mas não substituem a leitura exegética completa.\n")
            logging.info("Resultados de sumarização exibidos")
        except Exception as e:
            self.summary_text.insert(tk.END, f"Erro ao gerar resumos: {str(e)}\n")
            logging.error(f"Erro ao gerar resumos: {str(e)}")
            self.summary_text.insert(tk.END, "\n=== Palavras-chave (Fallback) ===\n")
            keywords = self.extract_keywords([text])
            top_keywords = [word for word, _ in keywords[:10]]
            self.summary_text.insert(tk.END, ", ".join(top_keywords) + "\n")
        finally:
            self.summary_text.config(state='disabled')

    def classify_text(self, verses_text):
        try:
            sample_texts = [
                ("Deus criou os céus e a terra", "positive"),
                ("A terra era sem forma e vazia", "negative"),
                ("Haja luz", "positive"),
                ("O dilúvio destruiu a terra", "negative"),
                ("Ama o teu próximo como a ti mesmo", "positive"),
                ("A cidade estava em caos", "negative"),
                ("E Deus viu que era bom", "positive"),
                ("Os homens se rebelaram contra Deus", "negative"),
                ("Este é o dia que o Senhor fez", "positive"),
                ("Eles estavam em silêncio", "neutral"),
                ("A criação foi concluída", "neutral"),
                ("O povo caminhava pelo deserto", "neutral"),
                ("Deus disse: Façamos o homem à nossa imagem", "positive"),
                ("O pecado entrou no mundo", "negative"),
                ("O Senhor é meu pastor", "positive"),
                ("A terra tremeu e houve escuridão", "negative"),
                ("Eu sou o caminho, a verdade e a vida", "positive"),
                ("O povo se desviou do caminho", "negative"),
                ("Bendito é o homem que confia no Senhor", "positive"),
                ("A ira de Deus foi revelada", "negative"),
                ("O reino de Deus está próximo", "positive"),
                ("Os ímpios serão destruídos", "negative")
            ]
            texts, labels = zip(*sample_texts)
            vectorizer = TfidfVectorizer(max_features=1000,
                                         stop_words=['o', 'a', 'os', 'as', 'um', 'uma', 'de', 'do', 'da', 'em', 'no',
                                                     'na', 'por', 'para', 'com', 'que', 'e', 'se'], min_df=1)
            X = vectorizer.fit_transform(texts)
            y = labels
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            classifier = MultinomialNB(alpha=1.0)
            classifier.fit(X_train, y_train)
            cv_scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            X_verses = vectorizer.transform(verses_text)
            predictions = classifier.predict(X_verses)
            self.classification_text.insert(tk.END, "Classificação de Sentimento dos Versículos\n\n", 'header')
            self.classification_text.insert(tk.END, "=== Estratégia de Rotulação ===\n")
            self.classification_text.insert(tk.END,
                                            "Os dados foram rotulados manualmente com base no tom emocional: 'positive' para versículos de criação, bênção ou esperança; 'negative' para destruição, pecado ou julgamento; 'neutral' para descrições factuais. A rotulação considera o contexto teológico.\n\n")
            self.classification_text.insert(tk.END, "=== Métricas do Modelo ===\n")
            self.classification_text.insert(tk.END, f"Acurácia: {accuracy:.2f}\n")
            self.classification_text.insert(tk.END, f"Precisão: {precision:.2f}\n")
            self.classification_text.insert(tk.END, f"Recall: {recall:.2f}\n")
            self.classification_text.insert(tk.END, f"F1-Score: {f1:.2f}\n")
            self.classification_text.insert(tk.END,
                                            f"Acurácia Média (Validação Cruzada): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}\n\n")
            self.classification_text.insert(tk.END, "=== Classificação dos Versículos ===\n")
            for i, (verse, pred) in enumerate(zip(verses_text, predictions), 1):
                self.classification_text.insert(tk.END, f"Versículo {i}: {verse[:50]}... -> {pred.capitalize()}\n")
            self.classification_text.insert(tk.END, "\n=== Análise Qualitativa ===\n")
            self.classification_text.insert(tk.END,
                                            "O modelo captura bem tons positivos em versículos de criação (ex.: 'Haja luz') e negativos em julgamentos (ex.: 'O dilúvio'). Versículos neutros podem ser mal classificados devido à ambiguidade. A expansão do conjunto de treinamento melhoraria a robustez.\n")
            self.classification_text.insert(tk.END, "\n=== Relevância ===\n")
            self.classification_text.insert(tk.END,
                                            "A classificação de sentimento ajuda a entender o tom emocional de capítulos, útil para estudos comparativos entre livros (ex.: Gênesis vs. Apocalipse). As métricas indicam confiabilidade moderada, mas o contexto bíblico exige cautela na interpretação.\n")
            logging.info("Resultados de classificação exibidos")
        except Exception as e:
            self.classification_text.insert(tk.END, f"Erro ao classificar texto: {str(e)}\n")
            logging.error(f"Erro ao classificar texto: {str(e)}")
        finally:
            self.classification_text.config(state='disabled')

    def deduce_chapter_theme(self, keywords, clusters, verses_text):
        try:
            self.theme_text.config(state='normal')
            self.theme_text.delete(1.0, tk.END)
            top_keywords = [word for word, _ in keywords[:5]]
            cluster_keywords = []
            if len(clusters) > 0:
                unique_clusters = set(clusters)
                for cluster_id in unique_clusters:
                    cluster_verses = [verses_text[i] for i, c in enumerate(clusters) if c == cluster_id]
                    cluster_text = " ".join(cluster_verses)
                    cluster_keywords.extend([word for word, _ in self.extract_keywords([cluster_text])[:2]])
            theme_words = list(set(top_keywords + cluster_keywords))
            if not theme_words:
                theme = "Indefinido (poucas palavras-chave relevantes)"
            else:
                theme = ", ".join(theme_words[:3])
                theme_map = {
                    ('deus', 'terra', 'luz'): "Criação divina e formação do mundo",
                    ('senhor', 'povo', 'lei'): "Relação entre Deus e o povo",
                    ('jesus', 'amor', 'vida'): "Ensino e missão de Jesus",
                    ('sabedoria', 'justo', 'caminho'): "Instrução e sabedoria divina"
                }
                for key_words, mapped_theme in theme_map.items():
                    if all(kw in [w.lower() for w in theme_words] for kw in key_words):
                        theme = mapped_theme
                        break
                else:
                    theme = f"Tema baseado em: {theme}"
            self.theme_text.insert(tk.END, "Tema Principal do Capítulo\n\n", 'header')
            self.theme_text.insert(tk.END, f"Tema Deduzido: {theme}\n\n")
            self.theme_text.insert(tk.END, "=== Palavras-Chave Principais ===\n")
            for word, score in keywords[:5]:
                self.theme_text.insert(tk.END, f"{word}: {score:.4f}\n")
            self.theme_text.insert(tk.END, "\n=== Relevância ===\n")
            self.theme_text.insert(tk.END,
                                   "A dedução do tema sintetiza a mensagem central do capítulo, combinando palavras-chave e agrupamentos semânticos. É útil para estudos bíblicos, permitindo identificar rapidamente o foco narrativo ou teológico, como a criação em Gênesis 1.\n")
            logging.info("Tema deduzido e exibido")
        except Exception as e:
            self.theme_text.insert(tk.END, f"Erro ao deduzir tema: {str(e)}\n")
            logging.error(f"Erro ao deduzir tema: {str(e)}")
        finally:
            self.theme_text.config(state='disabled')

    def guess_book(self, verses_text):
        try:
            self.guess_book_text.config(state='normal')
            self.guess_book_text.delete(1.0, tk.END)
            book_keywords = {
                'Gênesis': {
                    'keywords': ['criação', 'terra', 'céus', 'deus', 'luz', 'homem', 'criou', 'águas', 'firmamento',
                                 'noé', 'adão', 'eva', 'jardim', 'serpente', 'dilúvio'], 'weight': 2.0},
                'Êxodo': {
                    'keywords': ['moisés', 'egito', 'faraó', 'mar', 'povo', 'lei', 'libertação', 'sinai', 'tabernáculo',
                                 'pragas', 'êxodo', 'aliança'], 'weight': 2.0},
                'Salmos': {
                    'keywords': ['senhor', 'louvor', 'pastor', 'salmo', 'coração', 'alegria', 'justiça', 'misericórdia',
                                 'refúgio', 'adoração', 'harpa'], 'weight': 1.5},
                'Provérbios': {'keywords': ['sabedoria', 'justo', 'tolo', 'caminho', 'conhecimento', 'temor', 'senhor',
                                            'proverbio', 'instrução', 'retidão'], 'weight': 1.5},
                'João': {
                    'keywords': ['jesus', 'amor', 'vida', 'luz', 'mundo', 'discípulos', 'verdade', 'pai', 'evangelho',
                                 'milagre', 'ressurreição'], 'weight': 2.0},
                '1 Coríntios': {
                    'keywords': ['amor', 'igreja', 'espírito', 'corpo', 'graça', 'ressurreição', 'paulo', 'coríntios',
                                 'dons', 'unidade'], 'weight': 2.0}
            }
            results = []
            chapter_text = " ".join(verses_text).lower()
            chapter_words = re.findall(r'\b\w+\b', chapter_text)
            chapter_counter = Counter(chapter_words)
            chapter_scores = {book: 0.0 for book in book_keywords}
            for book, data in book_keywords.items():
                for keyword in data['keywords']:
                    if keyword in chapter_counter:
                        chapter_scores[book] += chapter_counter[keyword] * data['weight']
            max_score = max(chapter_scores.values())
            chapter_guess = max(chapter_scores, key=chapter_scores.get) if max_score > 0 else "Gênesis (padrão)"
            for i, verse in enumerate(verses_text, 1):
                words = re.findall(r'\b\w+\b', verse.lower())
                verse_counter = Counter(words)
                scores = {book: 0.0 for book in book_keywords}
                for book, data in book_keywords.items():
                    for keyword in data['keywords']:
                        if keyword in verse_counter:
                            scores[book] += verse_counter[keyword] * data['weight']
                max_score = max(scores.values())
                guessed_book = max(scores, key=scores.get) if max_score > 0 else chapter_guess
                results.append((i, verse[:50], guessed_book, scores[guessed_book]))
            self.guess_book_text.insert(tk.END, "Adivinhação do Livro dos Versículos\n\n", 'header')
            self.guess_book_text.insert(tk.END, "=== Resultados por Versículo ===\n")
            for verse_num, verse_text, book, score in results:
                self.guess_book_text.insert(tk.END,
                                            f"Versículo {verse_num}: {verse_text}... -> {book} (Pontuação: {score:.2f})\n")
            self.guess_book_text.insert(tk.END, f"\n=== Livro Provável do Capítulo ===\n")
            self.guess_book_text.insert(tk.END, f"{chapter_guess} (Pontuação: {chapter_scores[chapter_guess]:.2f})\n")
            self.guess_book_text.insert(tk.END, "\n=== Metodologia ===\n")
            self.guess_book_text.insert(tk.END,
                                        "A adivinhação usa um dicionário de palavras-chave específicas para cada livro, com pesos ajustados para livros narrativos (ex.: Gênesis, João) vs. poéticos (ex.: Salmos). A pontuação reflete a frequência e relevância das palavras. Um fallback garante que 'Indefinido' seja evitado, default para Gênesis.\n")
            self.guess_book_text.insert(tk.END, "\n=== Relevância ===\n")
            self.guess_book_text.insert(tk.END,
                                        "Identificar o livro de um versículo é útil para contextualizar citações bíblicas em estudos teológicos ou sermões. A abordagem é robusta para livros com vocabulário distinto, mas pode falhar em capítulos atípicos. A análise por capítulo melhora a precisão.\n")
            logging.info("Adivinhação de livro exibida")
        except Exception as e:
            self.guess_book_text.insert(tk.END, f"Erro ao adivinhar livro: {str(e)}\n")
            logging.error(f"Erro ao adivinhar livro: {str(e)}")
            self.guess_book_text.insert(tk.END, "\n=== Fallback ===\n")
            self.guess_book_text.insert(tk.END, "Livro: Gênesis (padrão devido a erro)\n")
        finally:
            self.guess_book_text.config(state='disabled')

    def save_to_file(self):
        try:
            self.result_text.config(state='normal')
            content = self.result_text.get(1.0, tk.END)
            self.result_text.config(state='disabled')
            if not content.strip():
                self.show_error("Nenhum conteúdo para salvar.")
                return
            os.makedirs('output', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            book = "biblia"
            chapter = "texto"
            book_match = re.search(r'LIVRO:\s*(\w+)', content, re.I)
            chapter_match = re.search(r'CAPÍTULO:\s*(\w+)', content, re.I)
            if book_match:
                book = book_match.group(1).lower()
            if chapter_match:
                chapter = chapter_match.group(1).lower()
            filename = f"output/{book}_cap{chapter}_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            self.update_status(f"Texto salvo com sucesso em {filename}", self.success_color)
            messagebox.showinfo("Sucesso", f"Arquivo salvo com sucesso em:\n{os.path.abspath(filename)}")
            logging.info(f"Texto salvo em {filename}")
        except Exception as e:
            self.show_error(f"Erro ao salvar arquivo: {str(e)}")
            logging.error(f"Erro ao salvar arquivo: {str(e)}")

    def copy_to_clipboard(self):
        try:
            self.result_text.config(state='normal')
            content = self.result_text.get(1.0, tk.END)
            self.result_text.config(state='disabled')
            if not content.strip():
                self.show_error("Nenhum conteúdo para copiar.")
                return
            pyperclip.copy(content)
            self.update_status("Texto copiado para a área de transferência!", self.success_color)
            logging.info("Texto copiado para a área de transferência")
        except Exception as e:
            self.show_error(f"Erro ao copiar para área de transferência: {str(e)}")
            logging.error(f"Erro ao copiar para área de transferência: {str(e)}")

    def clear_results(self):
        try:
            for text_widget in [self.result_text, self.keywords_text, self.cluster_text, self.entities_text,
                                self.summary_text, self.classification_text, self.theme_text, self.guess_book_text]:
                text_widget.config(state='normal')
                text_widget.delete(1.0, tk.END)
                text_widget.config(state='disabled')
            self.update_status("Pronto para começar.", "black")
            self.save_button.config(state='disabled')
            self.copy_button.config(state='disabled')
            logging.info("Resultados limpos")
        except Exception as e:
            self.show_error(f"Erro ao limpar resultados: {str(e)}")
            logging.error(f"Erro ao limpar resultados: {str(e)}")

    def update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)
        logging.info(f"Status atualizado: {message}")

    def show_error(self, message):
        self.update_status(f"Erro: {message}", self.error_color)
        messagebox.showerror("Erro", message)

    def show_about(self):
        about_text = """BibleScraper v3.6

Ferramenta profissional para extração e análise de textos bíblicos.

Recursos:
- Extração de textos de diversas traduções
- Identificação automática de versículos
- Análise de palavras-chave via TF-IDF
- Clusterização semântica de versículos
- Extração de entidades e atributos linguísticos
- Sumarização extrativa (LexRank, LSA)
- Classificação de sentimento supervisionada
- Dedução do tema do capítulo
- Adivinhação robusta do livro

Desenvolvido com Python, Tkinter, BeautifulSoup, sumy, scikit-learn.

© 2025 Extração de Textos Bíblicos"""
        messagebox.showinfo("Sobre", about_text)
        logging.info("Exibida janela 'Sobre'")

    def open_docs(self):
        docs_url = "https://github.com/bleedart/docs"
        try:
            webbrowser.open_new(docs_url)
            logging.info("Documentação aberta")
        except Exception as e:
            self.show_error("Não foi possível abrir a documentação.")
            logging.error(f"Erro ao abrir documentação: {str(e)}")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        window_width = 900
        window_height = 700
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        app = BibleScraperApp(root)
        root.mainloop()
        logging.info("Aplicação iniciada com sucesso")
    except Exception as e:
        logging.error(f"Erro ao iniciar aplicação: {str(e)}")
        raise
