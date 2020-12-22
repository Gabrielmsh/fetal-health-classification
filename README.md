# fetal-health-classification
Projeto de classificação multi-classes de exames de cardiotocografias

O projeto foi realizado utilizando python 3.8.2



A aquisição do conhecimento foi realizada a partir de 2126 exames de cardiotocografia (CTG),
a partir de ondas de ultrassom. O exame permite adquirir informações importantes a respeito da gravidez,
como frequência cardíaca fetal, movimento fetal, taquicardia e bradicardia fetal,
cuja interpretação dos resultados pelo profissional capacitado pode definir o melhor plano de ação
a ser tomado para minimizar a probabilidade de complicações da maternidade.
A base de dados está disponível a partir do link: https://www.kaggle.com/andrewmvd/fetal-health-classification



A representação do conhecimento desta base de dados consiste na disposição
dos valores obtidos através do exame CTG numa matriz de dimensões (2126,22), 
em que as linhas representam 1 exame (instância), e as 21 primeiras colunas 
representam os dados do exame como a frequência cardíaca fetal e contrações interinas. 
As colunas são comumente chamadas de atributos (features). A última coluna representa
as classificações de cada exame, às quais foram atribuídas valores numéricos para que 
o algoritmo pudesse interpretar a informação. Os valores atribuídos às classes foram 1, 2 e 3,
que representam, respectivamente, um exame normal, suspeito e patológico.



O conjunto total de atributos é composto por: frequência cardíaca média; 
taquicardias; movimento fetal; contrações uterinas; bradicardias leves; 
bradicardias severas; bradicardias prolongadas; variabilidade rápida anormal; 
valor médio das variabilidades rápidas; porcentagem do tempo com variabilidade anormal prolongadas; 
valor médio das variabilidades prolongadas; largura do histograma; valor mínimo do histograma; 
valor máximo do histograma; quantidade de picos; quantidade de zeros no histograma; moda do histograma; 
média do histograma; mediana do histograma; variância do histograma; tendência do histograma. 
