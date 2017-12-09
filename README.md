# ODE-Solver
Python implementation of various numerical methods for ordinary differential equations.

READ ME
Exercício de Implementações de Métodos - Hugo Lispector

INTRODUÇÃO 

Há duas pastas com programas referentes às implementações dos Métodos: "Programa Principal" e "Métodos Separados".

A pasta "Programa Principal" contém o arquivo "programa_completo.py". Nesse programa oferece-se uma interface simples, porém completa para, a partir de um problema de valor inicial inserido, plotar gráficos, visualizar em forma de tabela os pontos gerados e exibir estatisticamente os erros dos métodos numéricos. Tudo isso comparativamente entre um número arbitrário de métodos disponíveis a serem escolhidos pelo usuário. Para as opções referentes à análise de erro será pedido do usuário a inserção da solução analítica do problema dado inicialmente.

Na segunda pasta, "Métodos Separados", estão disponíveis os arquivos dos métodos separadamente para eventuais usos específicos. Cada um dos programas (Ex: Euler.py, Adams-Bashforth_5.py, Runge-Kutta_2.py, etc) recebe como entrada um problema de valor inicial e imprime o plot do gráfico referente, bem como uma tabela com os pontos gerados, através do método referenciado no nome do arquivo.


ANTES DA EXECUÇÃO

Para executar os programas de ambas as pastas será necessário ter instalado:

- Python 3, disponível em: https://www.python.org/downloads/

Também serão necessárias as instalações das seguintes bibliotecas de Python:

- prettytable, disponível em https://pypi.python.org/pypi/PTable/0.9.2
- numpy, disponível em https://www.scipy.org/scipylib/download.html
- matplotlib, disponível em https://matplotlib.org/users/installing.html
- parser, disponível internamente em Python
- math, disponível internamente em Python

Com isso, basta rodar o programa através do terminal ou de uma IDE com suporte a python, como PyCharm: https://www.jetbrains.com/pycharm/download/.

O funcionamento do programa foi testado e verificado nas plataformas: Ubuntu 16.04, Windows 10, macOS High Sierra e iOS 11 utilizando-se Python 3.


PASSO A PASSO PARA LINUX

#Instalar Python 3
sudo apt-get update
sudo apt-get install python3.6

#Instalar pip para baixar bibliotecas
sudo apt-get install python3-pip

#Instalar bibliotecas
sudo python3 -m pip install prettytable
sudo python3 -m pip install numpy
sudo python3 -m pip install matplotlib

#Instalar Pacote necessário adicional de Python 3
sudo apt-get install python3-tk

#Executar programa após entrar no diretório do arquivo.py
#Por exemplo: cd Desktop/Programa_Principal/
python3 programa_completo.py


DURANTE A EXECUÇÃO

Durante a execução do arquivo programa_completo.py, inicialmente será pedido do usuário digitar o número da opção deseada:

0 - Plotar Gráficos
1 - Analisar Erros
2 - Plotar Gráfico e Analisar Erros
3 - Sair do Programa

O usuário deve digitar um dos números exibidos. Por exemplo: "0".

Para a opção zero será pedido para inserir um problema de valor inicial na forma: x0, y0, xf, n, y'.

x0, y0 e xf são números reais e, caso o usuário deseje adicionar casas decimais, deverá usar o "." ponto, não a "," vírgula. Ex: "1", "0", "3.25"

Já o n é um número inteiro o qual deverá ser maior ou igual a 2 para os métodos Runge-Kutta e Adams-Moulton. E maior ou igual ao número 'm' da ordem dos métodos Adams-Bashforth 'm'.

A entrada y' é a expressão da função em termos de x e y e deve ser inserida conforme a sintaxe e funções matemáticas de python e da biblioteca "math". Para inserir a equação y'=sin(2*x)+4*y**2 deve-se apenas digitar o lado direito da equação: "sin(2*x)+4*y**2".

Operadores básicos de python: https://www.tutorialspoint.com/python/python_basic_operators.htm
Funções Matemáticas:
https://docs.python.org/3/library/math.html

Por fim, x0, y0, xf, n, y' devem ser inseridos separados por ", " (vírgula e espaço).

Exemplos de entrada de problemas de valor inicial:
0, 1, 1, 10, e**x
0, 1, 1, 30, 1-x+4*y
0, 0, 10, 30, cos(x)
0, 4, 10, 20, (x-y)/(x+y)
0, 1, 5, 20, cos(x)*y
0, 1, 1, 20, 2*(y**2+1)/(x**2+4)

Posteriormente essa tabela será exibida na tela e será pedido para o usuário digitar um ou mais índices, separados por " " (um espaço), dos métodos os quais ele deseja executar. 

Por exemplo, para exibir escolher Euler, Adams-Bashforth 8 e Runge-Kutta 4 deve-se digitar: "0 7 18". Note que não deve haver vírgulas.

+--------+---------------------------+
| Índice |           Método          |
+--------+---------------------------+
|   0    |    Euler (RK 1 ou AB 1)   |
|   1    |     Adams-Bashforth 2     |
|   2    |     Adams-Bashforth 3     |
|   3    |     Adams-Bashforth 4     |
|   4    |     Adams-Bashforth 5     |
|   5    |     Adams-Bashforth 6     |
|   6    |     Adams-Bashforth 7     |
|   7    |     Adams-Bashforth 8     |
|   8    |    Euler Inverso (AM 1)   |
|   9    |  Euler Aprimorado (AM 2)  |
|   10   |      Adams-Moulton 3      |
|   11   |      Adams-Moulton 4      |
|   12   |      Adams-Moulton 5      |
|   13   |      Adams-Moulton 6      |
|   14   |      Adams-Moulton 7      |
|   15   |      Adams-Moulton 8      |
|   16   | Euler Aprimorado 2 (RK 2) |
|   17   |       Runge-Kutta 3       |
|   18   |       Runge-Kutta 4       |
|   19   |       Runge-Kutta 5       |
|   20   |       Runge-Kutta 6       |
+--------+---------------------------+

Após isso o programa exibirá uma tabela com os pontos e abrirá uma janela com o(s) plot(s). Para continuar o uso do programa é preciso fechar a janela com o gráfico.
 
Caso o usuário escolha a opção 1 ou 2 será pedido, além do problema de valor inicial a solução analítica do problema de valor inicial dado como entrada. Essa solução deverá respeitar a mesma sintaxe do y' (descrito acima).

Exemplos de entrada da solução exata:
e**x
(4*x+19*e**(4*x)-3)/16
2**(1/2)*(x**2+8)**(1/2)-x
tan(arctan(x/2)+pi/4)

Para executar os arquivos dos métodos separados o processo é semelhante. Apenas é preciso inserir o problema de valor inicial com x0, y0, xf, n, y' separados por vírgulas, como descrito acima.










