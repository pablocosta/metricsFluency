import spacy
import numpy as np
from spacy_syllables import SpacySyllables
from numpy import mean
from math import log
from collections import Counter

nlp = spacy.load("pt_core_news_lg")
nlp.add_pipe("syllables")



article_tags = ['DET']
verb_tags = ['VERB']
auxiliary_verb_tags = ['AUX']
noun_tags = ['NOUN',
            'PNOUN']
adjective_tags = ['ADJ']
adverb_tags = ['ADV']
pronoun_tags = ['PRON']
numeral_tags = ['NUM']
conjunction_tags = ['CONJ', 'SCONJ']
preposition_tags = ['ADP',
                'ADPPRON']
interjection_tags = ['INTJ']

content_word_tags = verb_tags\
+ noun_tags\
+ adjective_tags\
+ adverb_tags

function_word_tags = article_tags\
+ preposition_tags\
+ pronoun_tags\
+ conjunction_tags\
+ interjection_tags

functions_as_noun_tags = ['NOUN', 'PROPN']
functions_as_adjective_tags = ['ADJ']

punctuation_tags = ['PUNCT']

particle_tags = ['PART']
symbol_tags = ['SYM']
unknown_tags = ['X']

fine_to_coarse = {'ADPPRON': 'ADP',
                'AUX': 'VERB',
                'PNOUN': 'NOUN'}

def loadPsicolinguistico():
    with open("./psicolinguistico.txt", 'r', encoding="utf-8") as fp:
        lines = fp.read().splitlines()[1:]
    lines = [i.split(',') for i in lines]
    dic = {}
    for i in lines:
        dic[i[0]] = dict()
        dic[i[0]]['concretude'] = float(i[3])
        dic[i[0]]['familiaridade'] = float(i[4])
        dic[i[0]]['imageabilidade'] = float(i[5])
        dic[i[0]]['idade_aquisicao'] = float(i[6])
    return dic


def getConcretude(text):
    """
        **Nome da Métrica**: concretude_mean

        **Interpretação**: quanto maior a média de concretude, menor a complexidade textual

        **Descrição da métrica**: média dos valores de concretude das palavras de conteúdo do texto

        **Definição dos termos que aparecem na descrição da métrica**: são consideradas palavras de conteúdo:
        substantivos, verbos, adjetivos e advérbios. Concretude é uma característica psicolinguística das palavras de
        conteúdo e significa o quanto a palavra pode ser traduzida por uma imagem na opinião dos falantes da língua. Os
        valores variam de 1 a 7 e quanto maior o valor, maior a concretude.

        **Forma de cálculo da métrica**: Identificam-se as palavras de conteúdo do texto. Em seguida, lematizam-se essas
        palavras, usando o DELAF, e procuram-se seus respectivos valores de concretude. Calcula-se a média desses
        valores (somam-se os valores e divide-se o resultado pela quantidade de palavras de conteúdo do texto presentes
        no repositório psicolinguístico).

        **Recursos de PLN utilizados durante o cálculo**: nlpnet, DELAF e repositório psicolinguístico

        **Limitações da métrica**: o repositório psicolinguístico tem 26.874 palavras e pode não conter todas as
        palavras procuradas. O repositório psicolinguístico foi construído automaticamente (e por isso, sujeito a
        vieses), usando como semente listas de palavras com seus respectivos valores de concretude, familiaridade,
        idade de aquisição e imageabilidade levantados junto a usuários da língua por psicolinguistas e psicólogos.

        **Crítica**: havia 8 métricas para cada característica psicolinguística e decidiu-se manter apenas 4, a fim de
        evitar redundância. Por esse motivo esta métrica foi comentada.

        **Projeto**: GUTEN
    """
    
    
    dictPsico = loadPsicolinguistico()
    #extrai as palavras de conteudo
    wordsContent = []
    for ele in nlp(text.lower()):
        if ele.pos_ in content_word_tags:
            wordsContent.append(ele.lemma_)
    
    values = []
    for word in wordsContent:
        if word in dictPsico:
            values.append(dictPsico[word]['concretude'])
    retorno = 0.0
    if len(values) > 0:
        retorno = float(np.mean(values))
    return np.nan_to_num(retorno)

print(getConcretude("O aumento de casos frustrou expectativas e fez as autoridades reverem estratégias."))


def getFlesh(text):
    """
        **Nome da Métrica**: flesch

        **Interpretação**: Índice de leiturabilidade de Flesch

        **Descrição da métrica**: O Índice de Legibilidade de Flesch busca uma correlação entre tamanhos médios de
        palavras e sentenças

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: 248.835 - [1.015 x (média de palavras por sentença)] - [84.6 x (Número de
        sílabas do texto / Número de palavras do texto)]

        **Recursos de PLN utilizados durante o cálculo**:

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: Coh-Metrix-Port

        **Teste**: Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça entre os itens do uniforme de
        alunos dos ensinos Fundamental e Médio nas escolas municipais, estaduais e federais. Ele defende a medida como
        forma de proteger crianças e adolescentes dos males provocados pelo excesso de exposição aos raios solares. Se
        a ideia for aprovada, os estudantes receberão dois conjuntos anuais, completados por calçado, meias, calça e
        camiseta.

        **Contagens**: 3 sentenças, 69 palavras, 160 sílabas. Médias: 23 palavras por sentença; 2,31 sílabas por palavra.

        **Resultado Esperado**: 248,835 – [1,015 x (23)] – [84,6 x (2,31)] => 248,835 – [23,345 – 195,43] = 29,316

        **Resultado Obtido**: 29,316

        **Status**: correto
    """

    #get mean words per setnence
    meanWords = []
    for sent in nlp(text).sents:
        mean = []
        for word in sent:
            if not(word.is_punct):
                mean.append(1)
        meanWords.append(np.sum(mean))

    data = [(token.text, token._.syllables_count) for token in nlp(text)]


    meanSyllables = np.sum([ele[1] for ele in data if ele[1]]) / np.sum(meanWords)
   

    return 248.835 - 1.015 * np.mean(meanWords) - 84.6 * meanSyllables

print(getFlesh("Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça entre os itens do uniforme de alunos dos ensinos Fundamental e Médio nas escolas municipais, estaduais e federais. Ele defende a medida como forma de proteger crianças e adolescentes dos males provocados pelo excesso de exposição aos raios solares. Se a ideia for aprovada, os estudantes receberão dois conjuntos anuais, completados por calçado, meias, calça e camiseta."))


def mattr(tokens, w=100):
    """Return the Moving Average Type-Token Ratio of a list of tokens.

    :text: TODO
    :returns: TODO

    """

    p = 0
    n = len(tokens)
    wft = dict()
    ttr = []

    if w > n:
        w = n

    for i in range(p, w):
        if tokens[i] in wft:
            wft[tokens[i]] += 1
        else:
            wft[tokens[i]] = 1
    ttr.append(len(wft.keys()) / sum(wft.values()))

    while n - w > p:
        p += 1
        if wft[tokens[p - 1]] == 1:
            wft.pop(tokens[p - 1])
        else:
            wft[tokens[p - 1]] -= 1
        if tokens[p + w - 1] in wft:
            wft[tokens[p + w - 1]] += 1
        else:
            wft[tokens[p + w - 1]] = 1
        ttr.append(len(wft.keys()) / sum(wft.values()))

    return mean(ttr)



def getTokenRation(text):
    """
        **Nome da Métrica**: ttr (type  token ratio)

        **Interpretação**: quanto maior o valor da métrica, mais complexo o texto

        **Descrição da métrica**: Proporção de palavras sem repetições (types) em relação ao total de palavras com
        repetições (tokens). Não se usa lematização das palavras, ou seja, cada flexão é computada como um type
        diferente.

        **Definição dos termos que aparecem na descrição da métrica**: Types são as palavras que ocorrem em um texto,
        descontando suas repetições. Tokens são todas as palavras que ocorrem em um texto, sem descontar as repetições.

        **Forma de cálculo da métrica**: contam-se todos os types e divide-se pela quantidade de tokens.

        **Recursos de PLN utilizados durante o cálculo**: tokenizador nltk

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: Coh-Metrix-Port

        **Teste**: O acessório polêmico entrou no projeto, de autoria do senador Cícero Lucena (PSDB-PB), graças a uma
        emenda aprovada na Comissão de Educação do Senado em outubro. Foi o senador Flávio Arns (PT-PR) quem sugeriu a
        inclusão da peça entre os itens do uniforme de alunos dos ensinos Fundamental e Médio nas escolas municipais,
        estaduais e federais. Ele defende a medida como forma de proteger crianças e adolescentes dos males provocados
        pelo excesso de exposição aos raios solares. Se a ideia for aprovada, os estudantes receberão dois conjuntos
        anuais, completados por calçado, meias, calça e camiseta.

        **Contagens**: 95 palavras, 58 das quais palavras de conteúdo, 57 types (só repete a palavra “senador”).

        **ResultadoO aumento de casos frustrou expectativas e fez as autoridades reverem estratégias. Obtido**: 0,821 (está computando todos os tokens e não só palavras de conteúdo)

        **Status**: correto
    """
    tokens = []

    for i in nlp(text):
        if not (i.pos_ in ["PUNCT"]):
            tokens.append(i.text.lower())
    return mattr(tokens)


print(getTokenRation("O acessório polêmico entrou no projeto, de autoria do senador Cícero Lucena (PSDB-PB), graças a uma emenda aprovada na Comissão de Educação do Senado em outubro. Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça entre os itens do uniforme de alunos dos ensinos Fundamental e Médio nas escolas municipais, estaduais e federais. Ele defende a medida como forma de proteger crianças e adolescentes dos males provocados pelo excesso de exposição aos raios solares. Se a ideia for aprovada, os estudantes receberão dois conjuntos anuais, completados por calçado, meias, calça e camiseta."))


def getBrunetIndex(text):
    """
        **Nome da Métrica**: brunet

        **Interpretação**: Os valores típicos da métrica variam entre 10 e 20, sendo que uma fala mais rica produz
        valores menores (THOMAS et al., 2005).

        **Descrição da métrica**: Estatística de Brunet é uma forma de type/token ratio que é menos sensível ao tamanho
         do texto.

        **Definição dos termos que aparecem na descrição da métrica**: quantidade de types considera palavras sem
        repetições e quantidade de tokens considera palavras com repetições.

        **Forma de cálculo da métrica**: W = N ** (V ** −0.165) quantidade de types elevada à quantidade de tokens
        elevada à constante -0,165.

        **Recursos de PLN utilizados durante o cálculo**:

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: Coh-Metrix-Dementia

        **Teste**: O acessório polêmico entrou no projeto, de autoria do senador Cícero Lucena (PSDB-PB), graças a uma
        emenda aprovada na Comissão de Educação do Senado em outubro. Foi o senador Flávio Arns (PT-PR) quem sugeriu a
        inclusão da peça entre os itens do uniforme de alunos dos ensinos Fundamental e Médio nas escolas municipais,
        estaduais e federais. Ele defende a medida como forma de proteger crianças e adolescentes dos males provocados
        pelo excesso de exposição aos raios solares. Se a ideia for aprovada, os estudantes receberão dois conjuntos
        anuais, completados por calçado, meias, calça e camiseta.

        **Contagens**: 95 tokens e 78 types

        **Resultado Esperado**: 9,199

        **Resultado Obtido**: 9,199

        **Status**: correto
    """
    #get all non ponctuation tokens
    tokens = []

    for i in nlp(text):
        if not (i.pos_ in ["PUNCT"]):
            tokens.append(i.text.lower())
    
    types = len(set(tokens))

    return len(tokens) ** types ** -0.165


print(getBrunetIndex("O acessório polêmico entrou no projeto, de autoria do senador Cícero Lucena (PSDB-PB), graças a uma emenda aprovada na Comissão de Educação do Senado em outubro. Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça entre os itens do uniforme de alunos dos ensinos Fundamental e Médio nas escolas municipais, estaduais e federais. Ele defende a medida como forma de proteger crianças e adolescentes dos males provocados pelo excesso de exposição aos raios solares. Se a ideia for aprovada, os estudantes receberão dois conjuntos anuais, completados por calçado, meias, calça e camiseta."))


def getHonoreStatistics(text):
    """
        **Nome da Métrica**: honore

        **Interpretação**: quanto mais alto o valor, mais rico o texto é lexicalmente, o que está associado a maior
        complexidade.

        **Descrição da métrica**: Estatística de Honoré

        **Definição dos termos que aparecem na descrição da métrica**: N é o número total de tokens, V_1 é o número de
        palavras do vocabulário que aparecem uma única vez, e V é o número de palavras lexicais. (HONORÉ, 1979; THOMAS
        et al., 2005):

        **Forma de cálculo da métrica**: R = 100 * logN / (1 - (V_1 / V))

        **Recursos de PLN utilizados durante o cálculo**:

        **Limitações da métrica**:  Descrição da métrica: Estatística de Horoné

        **Crítica**:

        **Projeto**: Coh-Metrix-Dementia

        **Teste**: O acessório polêmico entrou no projeto, de autoria do senador Cícero Lucena (PSDB-PB), graças a uma
        emenda aprovada na Comissão de Educação do Senado em outubro. Foi o senador Flávio Arns (PT-PR) quem sugeriu a
        inclusão da peça entre os itens do uniforme de alunos dos ensinos Fundamental e Médio nas escolas municipais,
        estaduais e federais. Ele defende a medida como forma de proteger crianças e adolescentes dos males provocados
        pelo excesso de exposição aos raios solares. Se a ideia for aprovada, os estudantes receberão dois conjuntos
        anuais, completados por calçado, meias, calça e camiseta.

        **Contagens**: N= 95 tokens, V_1 = 69 hapax legomena, V = 78 types

        **Resultado Esperado**: 100 * log95 / (1-(69/78) => (100 * 1,97772)/ (1- 0,885) => 197,772/0,115 => 1719,756

        **Resultado Obtido**: 1714,027

        **Status**: correto, considerando arredondamentos
    """

    tokens   = []
    for i in nlp(text):
        if not (i.pos_ in ["PUNCT"]):
            tokens.append(i.text.lower())

    
    
    types = len(set(tokens))

    counter = Counter(tokens)
    tokensv1 = [word for word, count in counter.items()
                           if count == 1]

    honoreIndex = 100 * log(len(tokens), 10) / (1 - len(tokensv1) / types)

    return honoreIndex

print(getHonoreStatistics("O acessório polêmico entrou no projeto, de autoria do senador Cícero Lucena (PSDB-PB), graças a uma emenda aprovada na Comissão de Educação do Senado em outubro. Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça entre os itens do uniforme de alunos dos ensinos Fundamental e Médio nas escolas municipais, estaduais e federais. Ele defende a medida como forma de proteger crianças e adolescentes dos males provocados pelo excesso de exposição aos raios solares. Se a ideia for aprovada, os estudantes receberão dois conjuntos anuais, completados por calçado, meias, calça e camiseta."))






