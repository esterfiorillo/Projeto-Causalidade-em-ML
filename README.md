<div align="center">

# Projeto Causalidade em Aprendizado de Máquina

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

<!-- <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a> -->

<!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.6+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.2-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a> -->


</div>

<br>

## Membros

- Ester Fiorillo 
- Rodrigo Andrade

## 📌  Introduction

O aprendizado profundo (Deep Learnign - DL) é um dos ramos da inteligência artificial que teve um crescimento exponencial nos últimos anos. A comunidade científica tem focado sua atenção na DL devido à sua versatilidade, alto desempenho, alta capacidade de generalização, usos multidisciplinares, entre muitas outras qualidades. Além disso, uma grande quantidade de dados médicos se tornaram acessíveis e o desenvolvimento de computadores mais potentes também fomentaram o interesse na área de IA aplicado à dados médicos.

A radiografia de tórax é o exame de imagem mais comum em todo o mundo, crítico para triagem, diagnóstico e tratamento de muitas doenças potencialmente fatais. Interpretação automatizada de radiografia de tórax no nível de radiologistas praticantes pode fornecer benefícios substanciais em muitos casos médicos, como por exemplo na melhoria na priorização de pacientes, decisão clínica, apoio à triagem em larga escala e elaboração de iniciativas de saúde da população global.

## Dataset - CheXpert

O CheXpert é um grande conjunto de dados de radiografias de tórax (224.316 radiografias de 65.240 pacientes) coletado entre outrubro de 2002 e julho de 2017, no hospital de Stanford. O dataset foi interpretado para presença de 14  tipos de enfermidades. As interpretações foram feitas inicialmente pelos relatórios de avaliação e em seguida pelo consenso de um grupo de 3 radiologiastas.

Os dados de validação consistem em 200 amostras selecionadas aleatoriamente no dataset. O dataset de teste consiste em 500 casos de 500 pacientes inéditos aos dados de treino e validação. Oito radiologistas certificados pelo conselho de medicina americano anotaram individualmente cada um dos estudos no conjunto de testes, classificando cada enfermidade em: (1) presente, (2) provável incerto,  (3) improvável incerto e (4) ausente. Suas anotações foram binarizadas de forma que todos os casos prováveis presentes e incertos sejam tratados como positivos e todos os casos improváveis incertos e ausentes sejam tratados como negativos. A maioria dos votos de 5 anotações do radiologista serviu como uma anotação VERDADE; as 3 anotações restantes do radiologista foram usadas para avaliar o desempenho dos radiologistas.

Além do dataset os autores experimentaram várias arquiteturas de rede neural convolucional, especificamente ResNet152, DenseNet121, Inception-v4 e SEResNeXt101, e descobriu que a arquitetura DenseNet121 produziu os melhores resultados. Este classificador pode ser utilizado nos estudos de causalidade (discutido a seguir). Os autores treinaram modelos que tomam como entrada uma radiografia de tórax de visão única e emitem a probabilidade de cada uma das 14 observações. Quando mais de uma visualização estiver disponível, os modelos retornaram a probabilidade máxima das observações ao longo as visualizações.

## Modelo de DL - Classificador CheXnet

A Chexnet é um modelo baseado em uma rede neural artificial convolucional de 121 layers, feita para a tarefa da classificação multilabel, ou seja, uma mesma amostra pode possuir mais de uma classe. No caso do problema da classificação de doenças pulmonares em imagens de Raio-X torácicas e do dataset CheXpert escolhido, a Chexnet poderá fazer a predição de uma imagem em até 13 classes de doenças.

A rede foi originalmente treinada em outro dataset de imagens de Raio-X torácicas, o Chest-X-Ray 14. Ele também é considerado um conjunto de dados para a tarefa de classificação de doenças multilabel. Para o trabalho proposto foi feita a escolha de retreinar o modelo usando outro dataset, no caso o Chexpert, devido a esse ter sido lançado alguns anos depois e possuir ainda mais amostras que o Chest-X-Ray 14.

Mais detalhes sobre a rede neural que será utilizada incluem dizer que ela consiste em uma rede densa, logo ela possui um bom desempenho em passar fluxos de informações e gradientes através do modelo, sendo assim sua otimização mais tratável. Outro ponto importante a respeito da arquitetura é que seu último layer completamente conectado foi substituído por um que possui apenas um output e depois foi feita uma operação sigmoid não linear.

Por fim, a respeito da metodologia de treino, é pretendido que para experimentos iniciais o protocolo de treino seja semelhante ao apresentado no paper da Chexnet. Portanto, os pesos da Chexnet serãp inicializados com os pesos de um modelo pré treinado no ImageNet e a rede será treinada end-to-end com o otimizador Adam, uma taxa de aprendizado de 0,001 e em mini batchs de tamanho 16.

## Métricas

Como o problema em questão trata-se de uma classificação multilabel, a métrica utilizada nesse trabalho será a predominante na literatura para esse tipo de problema. No caso, será usada a área abaixo da curva ROC (AUROC). Uma curva ROC mostra a compensação entre a taxa de verdadeiros positivos (TPR) e a taxa de falsos positivos (FPR) em diferentes limites de decisão.

AUROC portanto é uma métrica de desempenho para “discriminação”: informa sobre a capacidade do modelo de discriminar entre casos (exemplos positivos) e não casos (exemplos negativos).

## Explicabilidade

Técnicas de Aprendizado de Máquina se tornaram extremamente boas em generalizar dados independentes e com distribuição semelhante. No caso das imagens de raio X e do problema da classificação de doenças nessas imagens, os modelos estado da arte atingiram métricas que chegam a ultrapassar a performance de profissionais da saúda na mesma tarefa. 

Entretanto, há uma grande barreira no uso efetivo desses algoritmos na vida prática, especialmente quando envolve a área da saúde. A causa dessa barreira se dá pela falta de explicabilidade das predições dos modelos, ou seja, há uma grande dificuldade em saber os motivos pelos quais aquele modelo disse que um exame de raio X possui o diagnóstico de uma enfermidade. Essa falta de explicações gera uma falta de confiança nos modelos, fato que faz muitas pessoas preferirem diagnósticos feitos por médicos e não por máquinas, mesmo sabendo que a máquina pode ter mais chances de acertar.

Nesse contexto, surgiu a área de explicabilidade em Aprendizado de Máquina e ela tem como objetivo expor o porque de alguma predição ter sido feita por algum modelo. Uma forma de fazer isso é por meio de métodos chamados atribuição de features, que fazem um ranking de features, ou seja, representam as contribuições de cada feature para o output do modelo. Nesse trabalho, foram pesquisadas duas ferramentas que visam realizar essa tarefa de demonstrar quais features fizeram mais diferença para que um modelo chegasse a uma predição. Ambas serão utilizadas junto a CheXnet, de forma a localizar pixels da imagem que foram mais relevantes para que a rede tenha classificado alguma imagem de Raio-X com alguma doença. Mais detalhe sobre cada uma dessas ferramentas se encontram nas subseções abaixo.

## CXPlain

O CXPlain é um método de explicação em que é treinado separadamente um modelo supervisionado para explicação causal do modelo de predição. São usadas funções de influência causal para quantificar a influencia das features e de grupos de features na acurácia do modelo.

Para cada feature, é calculado o fator Ai que denota a influencia causal de adcionar aquela feature ao conjunto de features de entrada. Por fim, também são fornecidos intervalos de confiança que estimam o nível de incerteza associada a importancia de cada feature.

## LIME

LIME é uma biblioteca que se propõem a fornecer uma implementação concreta de modelos explicativos locais. Os códigos estão disponíveis em https://github.com/marcotcr/lime. 

Conforme descrito em \citet*{Molnar}, a ideia é bastante intuitiva. Primeiro, esqueça os dados de treinamento e imagine que você tem apenas o classificar onde pode inserir pontos de dados e obter as previsões do modelo. O LIME testa o que acontece com as previsões quando você fornece variações de seus dados para o classificador. A biblioteca gera um novo conjunto de dados que consiste em amostras perturbadas e as previsões correspondentes do Classificador. Nesse novo conjunto de dados, o LIME treina um modelo interpretável, que é ponderado pela proximidade das instâncias amostradas à instância de interesse. O modelo aprendido deve ser uma boa aproximação das previsões do modelo de aprendizado de máquina localmente, mas não precisa ser uma boa aproximação global. Esse tipo de precisão também é chamado de fidelidade local.

Matematicamente, modelos explicativos locais com restrição de interpretabilidade podem ser expressos da seguinte forma:

\begin{equation}
explic(x) = \underset{g \ \in \ G}{argmin} \ L(f,g,\pi_(x) +\omega(g)
\end{equation}

Onde $L$ é a função de custo, $x$ é a instância local, $f$ é o classificador original, $g$ é o modelo explicativo,  G é a família de possíveis explicações e $\pi_(x)$ é uma métrica de distância da amostra original para a amostra gerada pelo LIME.

O passo-a-passo para treinar o modelo explicativo local é : 
\begin{itemize}
\item Selecione sua instância de interesse para a qual você deseja ter uma explicação da previsão do seu classificador . 
\item Perturbe seu conjunto de dados e obtenha as previsões do classificador para esses novos pontos. 
\item Ponderar as novas amostras de acordo com sua proximidade com a instância de interesse.
\item Treine um modelo ponderado e interpretável no conjunto de dados com as variações. 
\item Explique a previsão interpretando o modelo local.
\end{itemize}

Pretendemos comparar esta biblioteca com a performance e explicações do CXPlain.

## Validação com profissional da saúde

Após treinar o modelo de classificação de doenças pulmonares Chexnet e utilizar as ferramentas de explicabilidade Lime e CXPlain, teremos as predições obtidas pelo modelo e mapas que indicam quais features foram mais relevantes para essas predições terem sido feitas. 

No caso, essas ferramentas nos retornam regiões da imagem que tiveram mais relevancia para o modelo. Com esses recursos, espera-se que o modelo tenha utilizado de informações como textura, tamanho, presença de nódulos, entre outras características dos pulmões para predizer alguma doença. Nesse sentido e tendo esses recursos em mãos, a última etapa do trabalho consistirá em mostrar esses recursos para uma médica que o grupo tem contato e pedir uma análise dos mapas de relevancia apontados pelo lime e pelo CXPlain. O objetivo é e fato determinar se a rede utilizou elementos consistentes dos pulmões para determinar as classes de doenças.
