# Detecção de Objetos com YOLOv11
## Introdução

O **YOLO (You Only Look Once)** é um algoritmo de detecção de objetos que utiliza uma abordagem de rede neural convolucional (CNN) para realizar a detecção em tempo real. 

1. **Arquitetura da Rede**: O YOLO usa uma única rede neural convolucional para a detecção de objetos. A imagem de entrada é dividida em uma grade ( S \times S ). Cada célula da grade é responsável por prever ( B ) caixas delimitadoras e suas respectivas probabilidades de classe.
    
2. **Predições**: Para cada célula da grade, o YOLO prevê:
    
    - **Caixas delimitadoras**: Coordenadas ( (x, y, w, h) ), onde ( (x, y) ) são as coordenadas do centro da caixa em relação à célula da grade, e ( (w, h) ) são a largura e altura da caixa.
    - **Confiança**: Um valor que indica a certeza de que a caixa contém um objeto e a precisão da caixa delimitadora.
    - **Probabilidades de classe**: As probabilidades de que o objeto pertence a cada uma das classes.
3. **Função de Perda**: A função de perda do YOLO é composta por várias partes:
    
    - **Erro de localização**: Mede a diferença entre as caixas delimitadoras previstas e as reais.
    - **Erro de confiança**: Avalia a precisão das previsões de confiança.
    - **Erro de classificação**: Calcula a diferença entre as probabilidades de classe previstas e as reais.
4. **Processamento em Tempo Real**: Devido à sua arquitetura unificada, o YOLO é extremamente rápido, capaz de processar imagens em tempo real, o que é uma grande vantagem para aplicações que exigem detecção rápida e precisa.
    
5. **Versões do YOLO**: Desde a sua introdução, várias versões do YOLO foram desenvolvidas (YOLOv2, YOLOv3, YOLOv4, etc.), cada uma trazendo melhorias em termos de precisão e velocidade.
    

O YOLO é amplamente utilizado em diversas aplicações, como vigilância, direção autônoma e sistemas de segurança, devido à sua capacidade de detectar múltiplos objetos em uma única imagem de forma eficiente e precisa.

## Pre-requisitos

Primeiro de tudo você vai precisar do python instalado (no meu caso estou utilizando a versão 3.12.7). Além disso também precisará do git e do vscode.

![[./assets/SNAG-0002.png]]

Instalar ou atualizar a biblioteca ultralytics do YOLO:

```
pip install ultralytics
```

```
pip install --upgrade ultralytics
```

> Também vamos usar a bibloteca opencv mas não é necessário instalá-la pois já vai como dependência do ultralytics que já instalamos no passo anterior.

Faça um clone desse projeto teste no github:

```
git clone https://github.com/libotti/Deteccao-de-Objetos-com-Python-YOLO-e-OpenCV.git
```

Abra o projeto com o Visual Studio Code

<< incluir foto do projeto no vscode >>

### Explicando o código

Começamos setando o modelo a ser utilizado. Vamos usar o "yolov10n" que é o modelo mais simples e leve. O modelo "n" (nano) é o mais performático mas também o menos preciso. Em contrapartida os demais modelos (small, medium, large e extra) necessitam de maior poder de processamento para serem treinados e cada caso de aplicação deve ser estudado com cuidado a fim de escolher o que melhor atenda o seu caso. 

![[assets/SNAG-0018.png]]

#### Definindo o Input

Aqui apresento 3 formas de fazer o input (comente as linhas que não for usar)

![[assets/SNAG-0005.png]]

1. Caso 1: definido o path do arquivo de video (linux ou wsl no windows)
2. Caso 2: definindo um path de arquivo de video no windows (atente-se usar para as barras invertidas "/")
3. definindo sua câmera como dispositivo de input (o índice 0 deve capturar seu primeiro dispositivo mas caso possua mais, verifique o índice correto a usar)

#### Capturando o video

Aqui abrimos o video é aberto e verificamos se não possui algum erro, alem de capturar as dimensões do video e fps

![[Pasted image 20241116235251.png]]

#### Definindo as classes para contagem

Agora dizemos quais classes vamos contar de acordo com as opções do dataset. Defina um índice, vários índices ou nenhum (para buscar tudo). 

![[assets/SNAG-0009.png]]

> [!NOTE] Nota
> Para saber o índice das classes do YOLO, veja na documentação em [COCO - Ultralytics YOLO Docs](https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml)
> 
> ![[assets/SNAG-0007.png]]

![[assets/SNAG-0017.png]]

#### Definindo regiões

Aqui definimos a região onde os objetos serão contados ao passar por ela. Pode ser definida como uma linha, um retângulo ou polígono.

![[assets/SNAG-0010.png]]

#### Definindo o output

Aqui definimos onde sera gerada a saída com o desenho nas imagens e o tracker do objeto (atente-se para passar o path com as barras invertidas se estiver no windows)

![[assets/SNAG-0011.png]]

> [!NOTE] Nota
> É possível também apenas visualizar sem fazer o output em arquivo.

#### Passando Argumentos para o Contador de Objetos

Aqui vamos definir os arqumentos de configuração. 

![[assets/SNAG-0012.png]]

Atente-se que algumas configurações exigirão mais hardware

![[assets/SNAG-0014.png]]


> [!NOTE] Nota 1
> Para visualizar a lista completa de parâmetros, acesse [Object Counting - Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/object-counting/#real-world-applications)
> 
![[assets/SNAG-0013.png]]

> [!NOTE] Nota 2
> Quanto ao modelo, existem varios disponíveis, e para entender seu custo de processamento, acesse [COCO - Ultralytics YOLO Docs](https://docs.ultralytics.com/datasets/detect/coco/)
> 
![[assets/SNAG-0003.png]]

#### Processamento

Enquanto o video estiver sendo lido, o processamento continua frame a frame (até o fim).

![[assets/SNAG-0019.png]]
#### Adicionando uma opção para Interromper o Processo

Caso esteja processando a camera ou um video longo, pode ser necessário adicionar uma forma de parar o processamento. Neste caso, sera a tecla "q".

![[assets/SNAG-0020.png]]

### Executando o Código

Ao executar o código, como habilitamos o argumento show= True, uma janela será aberta  mostrando um preview do processamento. Este é um processo pesado e você terá a impressão de estar assistindo em câmera lenta. 

#### Usando Um Arquivo de Vídeo Como Input

O modelo utilizado e os parâmetros informados vão afetar diretamente na performance desse processo. Não se assuste se durante esse procedimento sua CPU chegar a 100% de uso.  

![[assets/SNAG-0021.png]]

Observe a janela de output do Visual Studio Code. Ela está apresentando exatamente o que esta sendo localizado no frame.

![[assets/SNAG-0000.png]]

Quando o processo finalizar, observe o arquivo de output que você definiu.

![[assets/SNAG-0001.png]]!

#### Usando a Camera Principal 

Ajustei o input para usar a camera:

![[assets/SNAG-0015.png]]

Defini uma região com um poligono de 4 coordenadas:

![[assets/SNAG-0022.png]]

E fiz alguns testes usando o modelo yolo11n.pt (mais leve):

![[assets/SNAG-0008.png]]


> [!NOTE] Nota
> Apesar do nivel de acertos alto, o nível de erros tambem foi alto. Por exemplo, ele não reconheceu garfo e faca e ainda indicou que meu tabuleio de xadrez como uma pizza e um pote de planta como copo.
> 

Repetindo o mesmo teste com o modelo yolo11x.pt:

![[assets/SNAG-0023.png]]

Por hora é isso. No próximo desejo demonstrar como fazer para retreirar o yolo com uma classe nova.
