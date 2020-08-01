# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TP I : Descentes de Gradient
# 
# Ce TP vise à apporter les éléments nécessaires pour comprendre les implementations des descentes de gradients, du moins dans un premier cadre naïf. **C'est un *TP à trous* ; il s'agira de compléter ces trous pour un rendu au plus tard le dimanche 30 août 2020.**
# 
# Voici un aperçu des points abordés lors de ce TP.
# 
# - Un set de fonctions tests: comment se donne-t-on une fonction numérique? 
# - Calculer le gradient d'une fonction: coder une solution en dur ou calcul approché?
# - La classe GD: une classe qui sert d'interface pour toutes les descentes qu'on souhaite implémenter. 
# - La Descente de Gradient :
#     - Calcul du pas : pas constant, *Backtracking*
#     - Choix de norme : la classique, la norme $1$
#     - Sensibilité aux points initiaux
#     - Sensibilité aux nombres de conditionnement
#     - Accélération : Momentum, Nesterov, Adam.
# - Cas d'application classique: la régression linéaire.
# - La Méthode de Newton : 
#     - La classe MN
#     - Comparaison à la GD.
# - Face aux contraintes d'égalité.
# 
# %% [markdown]
# ## Attendus de rendu
# 
# Vous êtes invités à compléter ce TP et à le prolonger par le travail qui vous est suggéré. Votre rendu sera jugé à l'aune de
# 
# - votre capacité à produire des algorithmes valides, répondant à la question posée
# - l'étude effectuée concernant la sensibilité de vos algorithmes aux hyperparamètres / conditions initiales
# - l'analyse comparative proposée quant aux différentes implémentations suggérées 
# - les stress-tests auxquels vous aurez confrontés vos implémentations. 
# 
# On portera une attention particulière à la *généricité* de votre réponse ; tout comme cela vous est suggéré par la suite on attendra de vous d'appuyer vos affirmations par suffisamment de tests et une appréciation pour les limites de votre analyse.
# 
# Ce TP est à rendre par **groupes de 3** et exceptionnellement **2**. Il sera complété d'une soutenance portant sur l'ensemble des TPs.
# %% [markdown]
# # Au travail!

# %%
import warnings
import math
import numpy as np

# %% [markdown]
# ## Un set de fonctions tests
# 
# Il y a différentes facons de se donner une fonction numérique en machine :
# 
# - Symboliquement : on se donne une expression symbolique représentant des sommes, différences, produits, quotients et composées de fonctions usuelles.  Dans ce cadre, l'opération de dérivation est symbolique et les dérivées obtenues par ce biais sont exactes. Une opération d'évaluation est nécessaire pour passer de l'écriture symbolique d'une fonction au calcul de la valeur qu'elle prend en un point. Vous pouvez vous attarder sur la bibliothèque `sympy` sous python, pour plus de détails.
# 
# - Numériquement : on envisage une fonction au sens algorithmique du terme, la particularité étant qu'elle a des entrées `floats` et une sortie du même type. Des approximations des fonctions usuelles sont implémentées dans les bibliothèques `math` et `numpy` de python. Dans un soucis de vectorialisation des opérations mathématiques, c'est la seconde bibliothèque qui est utilisée. Évaluer une fonction en un point consiste à appeler celle-ci avec les bonnes entrées flottantes. La dérivée d'une fonction en ce sens est une fonction au même sens qui s'exrime pour une entrée donnée à l'aide d'appels à la fonction de départ. 
# 
# - Par approximation : on se préoccupe nullement de la valeur de la fonction en tout point, il nous suffit d'avoir une liste de couples (entrée, sortie) ou encore (point, image) composées de flottants. Dans ce cas on approche les dérivées par les tangentes des droites joignants deux points successifs. 
# 
# Dans ce TP notre choix s'arrête sur la plus intuitives des démarches ; l'aspect *fonction au sens informatique du terme*. Vous êtes invités, si vous le souhaitez, à explorer les autres représentations et éventuellement les comparer à celui-ci.
# %% [markdown]
# ### Structure de données
# 
# On va se donner un ensemble de familles de fonctions ; une famille de fonctions est donnée par une classe qui wrap les attributs:
# 
# - `name` : une `str` qui contient le nom de la fonction. 
# - `convex` : `bool` spécifiant si la famille de fonction est convexe ou non.
# - `dim` : `int` la dimension de l'espace de départ.
# - `nb_params` : nombre de paramètres de la famille.
# - `value` : une fonction qui prend en entrée un `numpy` array et renvoie un `float` en sortie.
# - `grad` : fonction gradient de `function`.
# - `hess` : fonction hessienne de `function`.
# - `HCN` : fonction nombre de conditionnement de la hessienne de `function`.
# 

# %%
class test_family():
    
    def __init__(self, name, value, dim, nb_params, convex=None, grad=None, hess=None, HCN=None):
        self.name = name
        self.value = value
        self.dim = dim
        self.nb_params = nb_params
        self.convex = convex
        self.grad = grad
        self.hess = hess
        self.HCN = HCN

# %% [markdown]
# ### Fonctions du set de test

# %%
from matplotlib import pyplot as plt
import seaborn as sns


# %%
sns.set_style("whitegrid")

# %% [markdown]
# 1. Définir des familles de fonctions *convexes* sur $\mathbb{R}$ ayant un nombre de conditionnement uniquement dépendant des paramètres de la famille.  

# %%
quad_I_d = {
    "name": "quadratic_1",
    "dim" : 1,
    "nb_params" : 1,
    "convex" : True,
    "value" : (lambda x, gamma: gamma*(x**2) + x + 1),
    "grad"  : (lambda x, gamma: 2*gamma*x + 1),
    "hess"  : (lambda x, gamma: 2*gamma),
    "HCN"   : (lambda x, gamma: 1 ) 
}

quad_I = test_family(**quad_I_d)


# %%
fig, ax = plt.subplots(figsize=(16, 9))
x = np.linspace(-10, 10, 400)
ax.set_ylim(-1, 50)
for gamma in range(5, 50, 5):
    ax.plot(x, quad_I.value(x, gamma), label="gamma: {}".format(gamma))
ax.set_title("Famille quadratiques en dimension 1")
ax.legend()


# %%
cubic_I_d = { 
    "name": "cubic_1",
    "dim" : 1,
    "nb_params" : 1,
    "convex" : False,
    "value" : (lambda x, gamma: x**3 + gamma*x**2 + x + 1),
    "grad"  : (lambda x, gamma: 3*x**2 + 2*gamma*x + 1),
    "hess"  : (lambda x, gamma: 6*x + 2*gamma),
    "HCN"   : (lambda x, gamma: 1 ) 
}

cubic_I = test_family(**cubic_I_d)


# %%
fig, ax = plt.subplots(figsize=(16, 9))
x = np.linspace(-10, 10, 400)
ax.set_ylim(-1, 100)
for gamma in range(0, 11, 1):
    ax.plot(x, cubic_I.value(x, gamma), label="gamma: {}".format(gamma))
ax.set_title("Famille cubique en dimension 1")
ax.legend()


# %%
exp_I_d = {
    "name": "exp_1",
    "dim" : 1,
    "nb_params" : 1,
    "convex" : True ,
    "value" : (lambda x, gamma: np.exp(gamma*x) + np.exp(-gamma*x)),
    "grad"  : (lambda x, gamma: gamma*(np.exp(gamma*x) - np.exp(-gamma*x))),
    "hess"  : (lambda x, gamma: (gamma**2)*(np.exp(gamma*x) + np.exp(-gamma*x))),
    "HCN"   : (lambda x, gamma: 1 )    
}

exp_I = test_family(**exp_I_d)


# %%
fig, ax = plt.subplots(figsize=(16, 9))
x = np.linspace(-10, 10, 400)
ax.set_ylim(-1, 100)
params = np.linspace(0, 5, 20)
for gamma in params:
    ax.plot(x, exp_I.value(x, gamma), label="gamma: {}".format(gamma))
ax.set_title("Famille en cosinus hyperbolique en dimension 1")
ax.legend()


# %%
multi_sinks_d = {
    "name": "multi_sinks_1",
    "dim" : 1,
    "nb_params" : 1,
    "convex" : False ,
    "value" : (lambda x, gamma: 20*np.cos(x**2) + (gamma * x**2)),
    "grad"  : (lambda x, gamma: np.sin(x) + 2*gamma*x),
    "hess"  : (lambda x, gamma: np.cos(x) + 2*gamma),
    "HCN"   : (lambda x, gamma: 1 )  
}

multi_sinks = test_family(**multi_sinks_d)


# %%
fig, ax = plt.subplots(figsize=(16, 9))
x = np.linspace(-10, 10, 400)
ax.set_ylim(-100, 1000)
for gamma in range(1, 11, 1):
    ax.plot(x, multi_sinks.value(x, gamma), label="gamma: {}".format(gamma))
ax.set_title("Famille multi-puits en dimension 1")
ax.legend()

# %% [markdown]
# 2. Faire de même avec des fonctions sur $\mathbb{R}^2$.

# %%
quad_II_d = {
    "name": "quadric_2",
    "dim" : 2,
    "nb_params" : 1,
    "convex" : True ,
    "value" : (lambda x, gamma: x[:, 0]**2 + gamma*x[:, 1]**2),
    "grad"  : (lambda x, gamma: np.array([2*x[:, 0], 2*gamma*x[:, 1]]).reshape(1, -1)),
    "hess"  : (lambda x, gamma: np.diag([2, 2*gamma])),
    "HCN"   : (lambda x, gamma: gamma)
}

quad_II = test_family(**quad_II_d)


# %%
quad_II_d["grad"](np.array([[5, 5]]), 1)


# %%
nb_pts, x_mi, x_ma, y_mi, y_ma = 200, -2, 2, -2, 2
fig, ax = plt.subplots(2, 1, figsize=(20, 40))
x, y = np.linspace(x_mi, x_ma, nb_pts), np.linspace(x_mi, x_ma, nb_pts)
X, Y = np.meshgrid(x, y)
## Reshaping X, Y for proper evaluation by input function
x_y = np.vstack([X.reshape(1, -1), Y.reshape(1, -1)]).reshape(1, 2, -1)
for i in [0, 1]:
    ax[i].set_xlim(x_mi, x_ma)
    ax[i].set_ylim(y_mi, y_ma)
ax[0].contour(X, Y, quad_II.value(x_y, 10).reshape(nb_pts, -1), 15)
ax[1].contour(X, Y, quad_II.value(x_y, 1).reshape(nb_pts, -1), 15)

# %% [markdown]
# 3. Construire des familles de fonctions tests de dimensions de plus en plus grandes en fonction du paramètre. 

# %%
'''cubic_II_d = {
    "name": "cubic_2",
    "dim" : 2,
    "nb_params" : 1,
    "convex" : False ,
    "value" : (lambda x, gamma: x[:, 0]**3 + gamma*x[:, 1]**3),
    "grad"  : (lambda x, gamma: np.array([3*x[:, 0]**2, 3*gamma*x[:, 1]**2]).reshape(1, -1)),
    "hess"  : (lambda x, gamma: np.diag([6*x[:, 0], 6*gamma*x[:, 1]])),
    "HCN"   : (lambda x, gamma: gamma)
}

cubic_II = test_family(**cubic_II_d)

exp_II_d = {
    "name": "exp_2",
    "dim" : 2,
    "nb_params" : 1,
    "convex" : False ,
    "value" : (lambda x, gamma: np.exp(gamma*x[:, 0]*x[:, 1]) + np.exp(-gamma*x[:, 0]*x[:, 1])),
    "grad"  : (lambda x, gamma: np.array(gamma*x[:, 1]*(np.exp(gamma*x[:, 0]*x[:, 1]) - np.exp(-gamma*x[:, 0]*x[:, 1])), gamma*x[:, 0]*(np.exp(gamma*x[:, 0]*x[:, 1]) - np.exp(-gamma*x[:, 0]*x[:, 1]))).reshape(1, -1)),
    "hess"  : (lambda x, gamma: (gamma**2)*(np.exp(gamma*x) + np.exp(-gamma*x))),
    "HCN"   : (lambda x, gamma: 1 )    
}

exp_II = test_family(**exp_II_d)

x^(3)+x^(2)y+xy^(2)+y^(3)+x^(2)+xy+y^(2)+x+y+1'''

# %% [markdown]
# 4. Construire une `test_list` contenant la liste des familles de fonctions tests crées jusque là.

# %%
test_list = [quad_I, cubic_I, exp_I, multi_sinks, quad_II]

# %% [markdown]
# ## Différencier une fonction
# 
# Pour calculer le gradient d'une fonction on a déjà besoin de savoir calculer la dérivée d'une fonction réelle. On s'intéresse à cette question dans cette section. Le point central est le fait qu'approcher numériquement la dérivée d'une fonction réelle est *prone* aux erreurs numériques ; on est souvent dans une meilleure posture lorsque l'on a une expression explicite pour le gradient d'une fonction.
# %% [markdown]
# ### L'approche naïve
# %% [markdown]
# On rappelle que pour une fonction numérique le nombre dérivée d'une fonction $f : \mathbb{R} \to \mathbb{R}$ en un point $x \in \mathbb{R}$ est donnée par la limite :
# 
# $$ f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h} $$
# 
# quand celle-ci existe. D'un point de vue numérique une assez petite valeur $h$ donne un quotient suffisamment proche du nombre dérivée que l'on cherche. Ainsi pour $h$ assez petit on approche $f'(x)$ par 
# 
# $$ f'(x) \simeq \frac{f(x + h) - f(x)}{h} .$$
# 
# L'erreur à l'approximation est en $o(1)$ en $0$. 
# %% [markdown]
# 1. Coder une fonction `partial_` qui calcule la dérivée partielle d'une fonction à plusieurs variables en utilisant la démarche précédente.

# %%
def partial_(f, x, i=0, dx=1e-6):
    """Computes i-th partial derivative of f at point x.
    
    Args:
        f: objective function.
        x: point at which partial derivative is computed.
        i: coordinate along which derivative is computed.
        dx: slack for finite difference.
        
    Output:
        (float)

    """
    x = x.reshape(1, -1)
    h = np.zeros(x.shape)
    h[0, i] = dx
    return (f(x + h) - f(x)) / dx

# %% [markdown]
# 2. Tester votre fonction sur un exmple dont vous connaissez la dérivée partielle et comparer vos résutlats.

# %%
partial_(lambda x: np.exp(x), np.array([100]))


# %%
np.exp(np.array([100]))

# %% [markdown]
# **La différence est *petite* ou *grande*?**
# %% [markdown]
# ### Une démarche un peu moins naïve
# %% [markdown]
# On peut en réalité faire un peu mieux ; le nombre dérivée est également la limite 
# 
# $$ f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x-h)}{2h} $$.
# 
# On approche donc $f'(x)$ par 
# 
# $$f'(x) \simeq \frac{f(x+h)-f(x-h)}{2h}$$.
# 
# La différence avec la première approximation résulte de l'écriture du DL du numérateur de cette seconde écriture (en supposant $f$ deux fois différentiable:
# 
# $$f(x + h) - f(x - h) = (f(x) + hf'(x) + \frac{h^2}{2}f''(x) + o(h^2)) - (f(x) - hf'(x) + \frac{h^2}{2} f''(x) + o(h^2))$$
# 
# Et donc:
# 
# $$f(x + h) - f(x - h) = 2hf'(x) + o(h^2)$$.
# 
# On trouve que l'erreur d'approximation dans le second cas est désormais $o(h)$ ; ce qui est supposé être une meilleure approximation que dans le premier cas.
# 
# **Remarque : si l'on souhaite faire un calcul précis de l'erreur dans l'approximation de la dérivée de $f$ en un point on s'y prendrait pas tout à fait comme cela, nous ne donnons ici qu'une vague idée de la différence entre les deux démarches précédentes.**
# %% [markdown]
# 1. Utiliser la démarche précédente pour approcher la dérivée partielle d'une fonction en un point. Cette fonction sera notée `partial`.

# %%
def partial(f, x, i=0, dx=1e-6):
    """Computes i-th partial derivative of f at point x.
    
    Args:
        f: objective function.
        x: point at which partial derivative is computed.
        i: coordinate along which derivative is computed.
        dx: slack for finite difference.
        
    Output:
        (float)

    """
    x = x.reshape(1, -1)
    h = np.zeros(x.shape)
    h[0, i] = dx
    return (f(x + h) - f(x - h)) / (2*dx)

# %% [markdown]
# 2. Comparer `partial` à `partial_` et à l'expression exacte de la dérivée partielle d'une fonction de votre choix. 

# %%
partial(lambda x: np.exp(x), np.array([100]))


# %%
np.exp(np.array([100]))

# %% [markdown]
# Il est possible de raffiner d'avantage l'estimation de la dérivée d'une fonction en un point. On se contente de ces deux premières approches simples. Pour plus de détail vous pouvez vous référer au cours d'Analyse numérique de J-P. Demailly.
# %% [markdown]
# 4. Effectuer des tests plus larges sur les divergences des différents schémas d'approximation précédents.
# %% [markdown]
# ### Le calcul du gradient
# %% [markdown]
# On sait déjà par la démarche précédente que l'approximation numérique de la dérivée partielle d'une fonction en un point sera difficilement meilleure qu'une expression exacte. Parfois on n'est pas en mesure de trouver facilement et à la main la dérivée partielle d'une fonction ; on se garde donc la possibilité de calculer numériquement le gradient d'une fonction.
# %% [markdown]
# - Écrire une fonction `gradient` qui renvoie le gradient d'une fonction en un point.

# %%
def gradient(f, x, dx=1e-6):
    """Computes gradient of f at point x.
    
    Args:
        f: objective function.
        x: point at which gradient is computed.
        dx: slack for finite difference of partial derivatives.
        
    Output:
        (ndarray) of size domain of f.
        
    """
    x = x.reshape(1, -1)
    dim = x.shape[1]
    return np.array([partial(f, x, i, dx) for i in range(dim)]).reshape(1, -1)


# %%
gradient(lambda x: x[0, 0]**5 + x[0, 1]*2, np.array([1, 10]))

# %% [markdown]
# ## La classe `GD`
# 
# La classe correspond à l'interface par laquelle on va implémenter et comparer les différentes variantes des descentes de gradients. L'objectif de cette section est de remplir les différents composants de calcul du principe qu'on a vu en cours. 

# %%
class GD():
    """Gradient Descent Object.
    
    Implements gradient descent aiming to compute optimal objective 
    value of convex functions and local optimal ones of none 
    convex functions.
    
    """    
    def __init__(self, ddir=None, rate=None, decay=None, tol=None, max_iter=None):
        """        
        Instantiates a GD object.
    
        Attributes:
        ddir: function computing descent direction.
        rate: function computing learning rate ; takes in
              - x (ndarray): current iterate
              - f (function): objective function
              - dir_x (ndarray) : descent direction
              - grad (function) : gradient function
              - nb_iter (int): number of iterations.
              - eta (float): hyper-parameter.
        decay: function computing decay.
        tol: slack tolerance.
        max_iter: upper bound on number of iterations.
    
        """
        self.ddir = ddir if ddir else (lambda x, f, grad, tol: - grad(x, f, tol))
        self.rate = rate if rate else (lambda x, f, dir_x, grad, nb_iter, eta : 0.01)
        self.decay = decay if decay else (lambda x, f, grad, tol: np.linalg.norm(grad(x, f, tol)))
        self.tol = tol if tol else 1e-6
        self.max_iter = max_iter if max_iter else 1000
    
    def __call__(self, x, f, grad, verbose=False):
        """Calling gradient descent object with specific starting point and optimal function.
        
        Args:
            x: initial starting point for descent.
            f: objective function of optimisation problem.
            grad: function outputing gradient value of function f at a given point.
        
        Output:
            (float) sub-optimal value up to tolerance if execution is proper.
            (ndarray) list of gradient descent iterates.
            (ndarray) list of graidents of iterates.
            (int) number of iterations.
            
        """
        x = x.reshape(1, -1)
        eta = 0
        n_iter = 0

        dir_x = self.ddir(x, f, grad, self.tol)
        rate_ = self.rate(x, f, dir_x, grad, n_iter, eta)
        delta_x = rate_ * dir_x
        iters, iters_dir = x, delta_x
        
        decay_x = self.decay(x, f, grad, self.tol)
        while decay_x > self.tol and n_iter < self.max_iter:
            ## Deciding on direction
            dir_x = self.ddir(x, f, grad, self.tol)
            rate_ = self.rate(x, f, dir_x, grad, n_iter, eta)
            delta_x =  rate_ * dir_x
            
            # Storing iterates
            iters = np.vstack([iters, x])
            
            ## Updating iterate
            x = x + delta_x
            
            ## Storing directions
            iters_dir = np.vstack([iters_dir, delta_x])
            
            ## Computing decay
            decay_x = self.decay(x, f, grad, self.tol)
            
            ## Updating iteration number
            n_iter += 1
            
        if decay_x > self.tol:
            warnings.warn("Decay didn't get under tolerance rate.", RuntimeWarning)
        
        if verbose:
            msg = " Iteration nu. = {}\n approx. = {}\n ob value = {}\n and decay = {}."
            print(msg.format(n_iter, x.flatten(), f(x), decay_x))
        
        return (x, iters, iters_dir, n_iter) 

    def momentum(self, x, f, grad, rate, alpha, verbose=False):
        x = x.reshape(1, -1)
        eta = 0
        n_iter = 0

        delta_x = - rate * grad(x, f, self.tol)
        iters, iters_dir = x, delta_x
        
        decay_x = self.decay(x, f, grad, self.tol)
        while decay_x > self.tol and n_iter < self.max_iter:
            delta_x =  alpha * delta_x - rate * grad(x, f, self.tol)
            
            # Storing iterates
            iters = np.vstack([iters, x])
            
            ## Updating iterate
            x = x + delta_x
            
            ## Storing directions
            iters_dir = np.vstack([iters_dir, delta_x])
            
            ## Computing decay
            decay_x = self.decay(x, f, grad, self.tol)
            
            ## Updating iteration number
            n_iter += 1
            
        if decay_x > self.tol:
            warnings.warn("Decay didn't get under tolerance rate.", RuntimeWarning)
        
        if verbose:
            msg = " Iteration nu. = {}\n approx. = {}\n ob value = {}\n and decay = {}."
            print(msg.format(n_iter, x.flatten(), f(x), decay_x))
        
        return (x, iters, iters_dir, n_iter)

    def nesterov(self, x, f, grad, rate, alpha, verbose=False):
        x = x.reshape(1, -1)
        eta = 0
        n_iter = 0

        delta_x = -rate * grad(x, f, self.tol)
        iters, iters_dir = x, delta_x
        
        decay_x = self.decay(x, f, grad, self.tol)
        while decay_x > self.tol and n_iter < self.max_iter:
            delta_x =  alpha * delta_x - rate * grad(x + alpha * delta_x, f, self.tol)
            
            # Storing iterates
            iters = np.vstack([iters, x])
            
            ## Updating iterate
            x = x + delta_x
            
            ## Storing directions
            iters_dir = np.vstack([iters_dir, delta_x])
            
            ## Computing decay
            decay_x = self.decay(x, f, grad, self.tol)
            
            ## Updating iteration number
            n_iter += 1
            
        if decay_x > self.tol:
            warnings.warn("Decay didn't get under tolerance rate.", RuntimeWarning)
        
        if verbose:
            msg = " Iteration nu. = {}\n approx. = {}\n ob value = {}\n and decay = {}."
            print(msg.format(n_iter, x.flatten(), f(x), decay_x))
        
        return (x, iters, iters_dir, n_iter)

    def adam(self, x, f, grad, rate, beta1, beta2, epsilon, verbose=False):
        x = x.reshape(1, -1)
        eta = 0
        n_iter = 1

        m = 0
        #mt = m / (1 - beta1 ** (n_iter + 1))
        v = 0
        #vt = v / (1 - beta2**(n_iter + 1))

        #delta_x = - rate * mt / (np.sqrt(vt) + epsilon)
        #iters, iters_dir = x, delta_x
        iters, iters_dir = x, 0
        
        decay_x = self.decay(x, f, grad, self.tol)
        while decay_x > self.tol and n_iter < self.max_iter:
            
            # Storing iterates
            iters = np.vstack([iters, x])
            
            ## Updating iterate
            m = beta1 * m + (1 - beta1) * grad(x, f, self.tol)
            mt = m / (1 - (beta1**n_iter))
            v = beta2 * v + (1 - beta2) * (grad(x, f, self.tol)**2)
            vt = v / (1 - (beta2**n_iter))
            delta_x = -(rate * mt) / (np.sqrt(vt) + epsilon)
            x = x + delta_x
            
            ## Storing directions
            iters_dir = np.vstack([iters_dir, delta_x])
            
            ## Computing decay
            decay_x = self.decay(x, f, grad, self.tol)
            
            ## Updating iteration number
            n_iter += 1
            
        if decay_x > self.tol:
            warnings.warn("Decay didn't get under tolerance rate.", RuntimeWarning)
        
        if verbose:
            msg = " Iteration nu. = {}\n approx. = {}\n ob value = {}\n and decay = {}."
            print(msg.format(n_iter, x.flatten(), f(x), decay_x))
        
        return (x, iters[1, ...], iters_dir[1, ...], n_iter)

    def newton(self, x, f, grad, hess, verbose=False):
        x = x.reshape(1, -1)
        eta = 0
        n_iter = 0

        delta_x = 0
        h = np.atleast_2d(hess(x, f, self.tol))
        g = np.atleast_2d(grad(x, f, self.tol))
        delta_x = - (np.linalg.inv(h) @ g.reshape((h.shape[1], -1))).reshape(x.shape)
        #if type(h) is np.ndarray and h.shape[0] > 1:
        #    delta_x = - np.linalg.inv(h).dot(g.reshape((h.shape[0], -1)))
        #else:
        #    delta_x = - (1 / h) * g

        iters, iters_dir = x, delta_x
        
        decay_x = self.decay(x, f, grad, self.tol)
        while decay_x > self.tol and n_iter < self.max_iter:
            
            # Storing iterates
            iters = np.vstack([iters, x])
            
            ## Updating iterate
            h = np.atleast_2d(hess(x, f, self.tol))
            g = np.atleast_2d(grad(x, f, self.tol))
            delta_x = - (np.linalg.inv(h) @ g.reshape((h.shape[1], -1))).reshape(x.shape)
            x = x + delta_x
            #if type(h) is np.ndarray and h.shape[0] > 1:
            #    x = x - (np.linalg.inv(h).dot(g.reshape((h.shape[1], -1)))).reshape((1, -1))
            #else:
            #    x = x - (1 / h) * g
            
            ## Storing directions
            iters_dir = np.vstack([iters_dir, delta_x])
            
            ## Computing decay
            decay_x = self.decay(x, f, grad, self.tol)
            
            ## Updating iteration number
            n_iter += 1
            
        if decay_x > self.tol:
            warnings.warn("Decay didn't get under tolerance rate.", RuntimeWarning)
        
        if verbose:
            msg = " Iteration nu. = {}\n approx. = {}\n ob value = {}\n and decay = {}."
            print(msg.format(n_iter, x.flatten(), f(x), decay_x))
        
        return (x, iters, iters_dir, n_iter)

# %% [markdown]
# ### GD dans le cas d'un pas constant
# %% [markdown]
# Pour être en mesure de faire nos premiers tests d'utilisation de `GD` il nous faut encore se préoccuper des quelques hyperparamètres à fixer. On commence en un premier temps par tester la descente de gradient classique à pas constants.
#  
# %% [markdown]
# 1. Mettre les valeurs par défaut pour les attributs de `GD` de facon à obtenir une descente de gradients classique à pas constant et une condition d'arrêt donnée par le fait que le gradient en un point tombe en-deça d'un certain seuil de tolérance.
# %% [markdown]
# *Fait pour vous dans la GD, plus haut.*
# %% [markdown]
# 2. Tester cette descente de gradient classique sur les différentes fonctions tests. Répertorier dans un tableau le point initial de chaque descente, le nombre d'itérations, le pas choisi et la valeur objectif obtenue.
# %% [markdown]
# Cherchons d'abord à manipuler la classe qu'on vient de construire

# %%
GD_default = GD()


# %%
op_pt, iters, iters_dir, n_iter = GD_default(np.array([10]), (lambda x : quad_I.value (x, 2)), 
                                             (lambda x, f, tol : quad_I.grad (x, 2)), True)

# %% [markdown]
# On peut visualiser dans lors de l'optimisation de problèmes de petites dimensions les itérés de la descente. `matplotlib` est là pour  vous aider dans votre démarche.

# %%
fig, ax = plt.subplots(figsize=(16, 10))
f = lambda x: quad_I.value (x, 1)
x = np.linspace(-10, 10, 400)
ax.set_ylim(-50, 400)
ax.plot(x, f(x), "k-", label="Objective function")
# reshaping iters for input
iters_reshape = iters.reshape(iters.shape[0], 1)
ax.plot(iters_reshape, f(iters_reshape), 'r.', label="Iterates")
ax.quiver(iters_reshape, f(iters_reshape), iters_dir, -1, 
          color='r', scale=None, width=0.002, headwidth=5, headlength=10)
ax.set_title("Gradient Descent in dimensin 1.")
ax.legend()

# %% [markdown]
# On se contente ici de travailler avec les fonctions tests de dimension $1$. **À vous de généraliser.**

# %%
test_list_d1 = [test_family for test_family in test_list if test_family.dim == 1]


# %%
init_pt = np.array([1])


# %%
gamma = 1


# %%
cst_rate = 0.02


# %%
import pandas as pd


# %%
GD_test_summary = pd.DataFrame(columns=["init_pt", "rate", "optimal_pt", "nb_iter"])
GD_list = {}
for t_function in test_list_d1:
    idt = 0
    GD_list[t_function.name] = GD(rate = lambda x, f, dir_x, grad, nb_iter, eta : cst_rate)
    opt_pt, _, _, n_iter = GD_list[t_function.name](init_pt, 
                                                    (lambda x: t_function.value (x, gamma)), 
                                                    (lambda x, f, tol : t_function.grad(x, gamma)))
    naming_format = "{}_{}".format(t_function.name, idt)
    GD_test_summary.loc[naming_format] = pd.Series({"init_pt": init_pt, 
                                                     "rate": GD_list[t_function.name].rate (*[None]*6), 
                                                     "optimal_pt": opt_pt, 
                                                     "nb_iter": n_iter})
    idt += 1
GD_test_summary

# %% [markdown]
# 3. Étudier pour un sous-ensemble de votre choix la relation entre le pas de la descente et le nombre d'itérations. Prendre soins de bien vérifier que vous avez convergence.

# %%
init_pt = np.array([10])


# %%
gammas = np.linspace(1, 10, 10)


# %%
cst_rates = np.linspace(0.01, 0.5, 200)


# %%
GD_rate_iter = pd.DataFrame(columns=["gamma", "rate", "optimal_pt", "optimal_val", "nb_iter"])
GD_list_ri = {}
for t_function in test_list_d1:
    for gamma in gammas:
        idt = 0
        for cst_rate in cst_rates:
            GD_list_ri[t_function.name] = GD(rate = lambda x, f, dir_x, grad, nb_iter, eta : cst_rate)
            opt_pt, _, _, n_iter = GD_list_ri[t_function.name](init_pt, 
                                                               (lambda x: t_function.value (x, gamma)), 
                                                               (lambda x, f, tol : t_function.grad(x, gamma)))
            naming_format = "{}_{}_{}".format(t_function.name, idt, gamma)
            GD_rate_iter.loc[naming_format] = pd.Series({"gamma": gamma, 
                                                         "rate": GD_list_ri[t_function.name].rate (*[None]*6), 
                                                         "optimal_pt": opt_pt,
                                                         "optimal_val": t_function.value(opt_pt, gamma),
                                                         "nb_iter": n_iter})
            idt += 1
GD_rate_iter


# %%
list_of_test_types = ["quadratic", "cubic", "exp", "multisinks"]


# %%
nb_figures = 2*len(gammas)
fig, ax = plt.subplots(nb_figures, 1, figsize=(16, 5*nb_figures))
selection = GD_rate_iter.index.str.contains("quadratic", regex=False)
i = 0
for gamma in gammas:
    df_selected = GD_rate_iter[(selection) & (GD_rate_iter["gamma"] == gamma)]
    ax[i].plot(df_selected["rate"], df_selected["nb_iter"])
    ax[i].set_xlim(0, 0.5)
    ax[i+1].plot(df_selected["rate"], df_selected["optimal_val"], "r")
    ax[i+1].set_ylim(-1, 100)
    ax[i+1].set_xlim(0, 0.5)
    ax[i].set_title("Gradient Descent in dimensin 1 for gamma = {}.".format(gamma))
    ax[i].legend()
    i += 2

# %% [markdown]
# 4. Étudier la relation entre la vitesse de convergence et le point initial de votre descente dans le cas des fonctions numériques.

# %%
init_points = np.linspace(-100, 100, 20)
gammas = 1
cst_rate = 0.02


# %%
GD_rate_iter = pd.DataFrame(columns=["function_name", "init_point", "nb_iter"])
GD_list_ri = {}
for t_function in test_list_d1:
    for init_point in init_points:
        GD_list_ri[t_function.name] = GD(rate = lambda x, f, dir_x, grad, nb_iter, eta : cst_rate)
        opt_pt, _, _, n_iter = GD_list_ri[t_function.name](init_point, 
                                                            (lambda x: t_function.value (x, gamma)), 
                                                            (lambda x, f, tol : t_function.grad(x, gamma)))
        #naming_format = "{}".format(t_function.name)
        #GD_rate_iter.loc[naming_format] = pd.Series({"init_point": init_point, "nb_iter": n_iter})
        GD_rate_iter = GD_rate_iter.append({"function_name": t_function.name, "init_point": init_point, "nb_iter": n_iter}, ignore_index=True)
GD_rate_iter.set_index("function_name")
GD_rate_iter


# %%
nb_figures = len(test_list_d1)
fig, ax = plt.subplots(nb_figures, 1, figsize=(16, 5*nb_figures))
i = 0
for function in test_list_d1:
    #selection = GD_rate_iter.index.str.contains(function_name, regex=False)
    #df_selected = GD_rate_iter[function_name]
    x = GD_rate_iter.groupby("function_name")
    x = x.get_group(function.name)
    x = x["init_point"].values
    y = GD_rate_iter.groupby("function_name")
    y = y.get_group(function.name)
    y = y["nb_iter"].values
    ax[i].plot(x, y)
    ax[i].set_xlim(-100, 100)
    ax[i].set_title("Gradient Descent in dimensin 1 for function {}.".format(function.name))
    ax[i].set_xlabel("starting point")
    ax[i].set_ylabel("number of iterations")
    ax[i].legend()
    i += 1
fig.tight_layout(pad=3.0)

# %% [markdown]
# ### GD dans le cas de *backtracking*
# %% [markdown]
# Vous devriez avoir constatés que le choix du pas de descente dans le cas constant est crucial pour garantir la convergence de l'algorithme de descente. Dans cette section on s'intéresse à un calcul adaptatif du pas de descente qui permet de mieux garantir la convergence de notre algo. Le désavatage est le temps que prend désormais chaque itération pour s'exécuter.
# %% [markdown]
# 1. Écrire une fonction `backtracking` qui permet de calculer le pas par *backtracking* à une itération donnée. Pour rappel le *backtracking* a deux hyper-paramètre $\alpha$ et $\beta$ respectivement mis par défaut à $0.01$ et $0.08$.

# %%
class backtracking():
    
    def __init__(self, alpha=0.01, beta=0.8, max_iter=100):
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        
    def __call__(self, x, f, dir_x, grad, n_iter, eta, tol=1e-6):
        n_while = 0
        t = 1
        x = x.reshape(1, -1)
        grad_f = grad(x, f, tol)
        while f(x + t * dir_x) > f(x) + self.alpha * t * grad_f.dot(dir_x.T) and n_while < self.max_iter:
            t = self.beta * t
            n_while += 1
        return t

# %% [markdown]
# 2. Comparer la GD par backtracking aux tests effectués jusqu'à présent dans le cas d'une GD à pas constant (à vous de réfléchir à ce que vous voulez comparer).

# %%
bt_rate = backtracking()


# %%
gammas = np.linspace(1, 10, 10)


# %%
GD_ri_bt = pd.DataFrame(columns=["gamma", "rate", "optimal_pt", "optimal_val", "nb_iter"])
GD_list_bt = {}
for t_function in test_list_d1:
    for gamma in gammas:
        GD_list_bt[t_function.name] = GD(rate = bt_rate)
        opt_pt, _, _, n_iter = GD_list_bt[t_function.name](init_pt, 
                                                           (lambda x: t_function.value (x, gamma)), 
                                                           (lambda x, f, tol : t_function.grad(x, gamma)))
        naming_format = "{}_{}".format(t_function.name, gamma)
        GD_ri_bt.loc[naming_format] = pd.Series({ "gamma": gamma, 
                                                  "rate" : "backtracking",
                                                  "optimal_pt": opt_pt,
                                                  "optimal_val": t_function.value(opt_pt, gamma),
                                                  "nb_iter": n_iter })
GD_ri_bt

# %% [markdown]
# ### Changer de direction de descente
# 
# On a explicité en cours trois types de descentes : 
# 
#  - La déscente standard : la direction de descente est celle du gradient, ce qu'on vient de regarder.
#  - La déscente de plus forte pente dans le cas de la norme $\ell_1$ : la direction de descente suit le vecteur de la base canonique de plus grande dérivée partielle en valeur absolue.
#  - L'algorithme de Newton où il s'agit de calculer l'inverse des hessiennes au point courant. 
#  
# *L'algorithme de Newton sera abordé dans une section à part.* On va donc se concentrer pour commencer sur le cas de descente en norme $\ell_1$. 
#  
# %% [markdown]
# 1. Écrire une fonction `dsgd` qui calcule la direction de descente de plus forte pente dans le cas de la norme $\ell_1$. 

# %%
def dsgd(x, f, grad, tol):
    x = x.reshape(1, -1)
    dim = x.shape[1]
    sdescent = np.array([0]*dim, dtype=float).reshape(1, -1)
    grad_f = grad(x, f, tol)
    abs_grad = np.abs(grad_f)
    imax_diff = np.argmax(abs_grad) 
    sdescent[0, imax_diff] = grad_f[0, imax_diff]
    return (- sdescent)      

# %% [markdown]
# 2. Comparer la descente de gradient $\ell_1$ à la descente classique.
# %% [markdown]
# Effectuer une telle comparaison en dimension $1$ n'a pas de sens (pourquoi?). On fait donc une partie de l'étude précédente en dimension $2$ pour avoir un référentiel de comparaison.

# %%
test_list_d2 = [test_family for test_family in test_list if test_family.dim == 2]


# %%
init_pt_d2 = np.array([[2, 2]])


# %%
GD_ri_2 = pd.DataFrame(columns=["gamma", "rate", "optimal_pt", "optimal_val", "nb_iter", "iters", "iters_dir"])
GD_list_2 = {}
for t_function in test_list_d2:
    for gamma in gammas:
        GD_list_2[t_function.name] = GD(rate = lambda x, f, dir_x, grad, nb_iter, eta : 0.1)
        opt_pt, iters, iters_dir, n_iter = GD_list_2[t_function.name](init_pt_d2, 
                                                     (lambda x: t_function.value (x, gamma)), 
                                                     (lambda x, f, tol : t_function.grad(x, gamma)))
        naming_format = "{}_{}".format(t_function.name, gamma)
        GD_ri_2.loc[naming_format] = pd.Series({ "gamma": gamma, 
                                                 "rate" : 0.01,
                                                 "optimal_pt": opt_pt,
                                                 "optimal_val": t_function.value(opt_pt, gamma),
                                                 "nb_iter": n_iter,
                                                 "iters": iters,
                                                 "iters_dir": iters_dir})
GD_ri_2.loc[:, "gamma": "nb_iter"]


# %%
nb_pts, x_mi, x_ma, y_mi, y_ma = 200, -2, 2, -2, 2
fig, ax = plt.subplots(2, 1, figsize=(15, 30))
x, y = np.linspace(x_mi, x_ma, nb_pts), np.linspace(x_mi, x_ma, nb_pts)
X, Y = np.meshgrid(x, y)
## Reshaping X, Y for proper evaluation by input function
x_y = np.vstack([X.reshape(1, -1), Y.reshape(1, -1)]).reshape(1, 2, -1)
## Plotting level curves for extreme cases
extreme_cases = [gammas[0], gammas[-1]]
for i in range(len(extreme_cases)):
    ax[i].set_xlim(x_mi, x_ma)
    ax[i].set_ylim(y_mi, y_ma)
    ax[i].contour(X, Y, test_list_d2[0].value(x_y, gammas[i]).reshape(nb_pts, -1), 15)
    ax[i].quiver(GD_ri_2.loc["quadric_2_{}".format(gammas[i]), "iters"][:, 0], 
                 GD_ri_2.loc["quadric_2_{}".format(gammas[i]), "iters"][:, 1], 
                 GD_ri_2.loc["quadric_2_{}".format(gammas[i]), "iters_dir"][:, 0], 
                 GD_ri_2.loc["quadric_2_{}".format(gammas[i]), "iters_dir"][:, 1],
                 color='b', scale=6, width=0.002, headwidth=4, headlength=5)
    ax[i].plot(GD_ri_2.loc["quadric_2_{}".format(gammas[i]), "iters"][:, 0], 
               GD_ri_2.loc["quadric_2_{}".format(gammas[i]), "iters"][:, 1],
               'b.')


# %%
GD_ri_1 = pd.DataFrame(columns=["gamma", "rate", "optimal_pt", "optimal_val", "nb_iter", "iters", "iters_dir"])
GD_list_1 = {}
for t_function in test_list_d2:
    for gamma in gammas:
        GD_list_1[t_function.name] = GD(ddir = dsgd, 
                                        rate = lambda x, f, dir_x, grad, nb_iter, eta : 0.1)
        opt_pt, iters, iters_dir, n_iter = GD_list_1[t_function.name](init_pt_d2, 
                                                     (lambda x: t_function.value (x, gamma)), 
                                                     (lambda x, f, tol : t_function.grad(x, gamma)))
        naming_format = "{}_{}".format(t_function.name, gamma)
        GD_ri_1.loc[naming_format] = pd.Series({ "gamma": gamma, 
                                                 "rate" : 0.01,
                                                 "optimal_pt": opt_pt,
                                                 "optimal_val": t_function.value(opt_pt, gamma),
                                                 "nb_iter": n_iter,
                                                 "iters": iters,
                                                 "iters_dir": iters_dir})
GD_ri_1.loc[:, "gamma": "nb_iter"]


# %%
nb_pts, x_mi, x_ma, y_mi, y_ma = 200, -2, 2, -2, 2
fig, ax = plt.subplots(2, 1, figsize=(15, 30))
x, y = np.linspace(x_mi, x_ma, nb_pts), np.linspace(x_mi, x_ma, nb_pts)
X, Y = np.meshgrid(x, y)
## Reshaping X, Y for proper evaluation by input function
x_y = np.vstack([X.reshape(1, -1), Y.reshape(1, -1)]).reshape(1, 2, -1)
## Plotting level curves for extreme cases
extreme_cases = [gammas[0], gammas[-1]]
for i in range(len(extreme_cases)):
    ax[i].set_xlim(x_mi, x_ma)
    ax[i].set_ylim(y_mi, y_ma)
    ax[i].contour(X, Y, test_list_d2[0].value(x_y, gammas[i]).reshape(nb_pts, -1), 15)
    ax[i].quiver(GD_ri_1.loc["quadric_2_{}".format(gammas[i]), "iters"][:, 0], 
                 GD_ri_1.loc["quadric_2_{}".format(gammas[i]), "iters"][:, 1], 
                 GD_ri_1.loc["quadric_2_{}".format(gammas[i]), "iters_dir"][:, 0], 
                 GD_ri_1.loc["quadric_2_{}".format(gammas[i]), "iters_dir"][:, 1],
                 color='b', scale=6, width=0.002, headwidth=4, headlength=5)
    ax[i].plot(GD_ri_1.loc["quadric_2_{}".format(gammas[i]), "iters"][:, 0], 
               GD_ri_1.loc["quadric_2_{}".format(gammas[i]), "iters"][:, 1],
               'b.')

# %% [markdown]
# On en vient désormais à comparer ces deux algos. Vu que les mise-à-jour internes semble demander le même nombre d'itérations il est raisonnable de s'intéresser en premier temps au nombre d'itérations.

# %%
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 11)
ax.set_ylim(-1, 1000)
ax.plot(GD_ri_2["gamma"], GD_ri_2["nb_iter"], label="l2")
ax.plot(GD_ri_1["gamma"], GD_ri_1["nb_iter"], "r", label="l1")
ax.set_title("Comparing number of iterates of l1 and l2 gradient descents against gamma values")
ax.legend()

# %% [markdown]
# ## Sensibilité de GD aux conditionnements de la hessienne
# %% [markdown]
# On s'intéresse dans cette section au comportement de la descente de gradients vis-à-vis de la géométrie locale de l'itéré courant. Cette étude va vous permettre de comprendre en quoi le manque de symétrie locale des fonctions objectifs rend plus lente la convergence de la descente de gradient.
# %% [markdown]
# 1. En utilisant la GD classique à pas constant tracer dans les cas de nombres de conditionnements qui ne dépendent que des paramètres de vos familles tests ce nombre contre le nombre d'itérations de la GD.

# %%
df = pd.DataFrame(columns=["family", "gamma", "iterations"])
gammas = np.arange(1, 11)
gd = GD()
for gamma in gammas:
    opt_pt, iters, iters_dir, n_iter = gd(np.array([[2, 2]]), (lambda x: quad_II.value (x, gamma)), (lambda x, f, tol : quad_II.grad(x, gamma)))
    df = df.append({"family": quad_II.name,"gamma": gamma, "iterations": n_iter}, ignore_index=True)
df = df.set_index("family")

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(df["gamma"].values, df["iterations"].values, label="vanilla")
ax.set_title("HCN against number of iterations for vanilla gradient descent")
ax.set_xlabel("HCN")
ax.set_ylabel("iterations")
ax.legend()

# %% [markdown]
# 2. Faire de même avec le cas de *backtracking*

# %%
bt_rate = backtracking()
df = pd.DataFrame(columns=["family", "gamma", "iterations"])
gammas = np.arange(1, 11)
gd = GD(rate = bt_rate)
for gamma in gammas:
    opt_pt, iters, iters_dir, n_iter = gd(np.array([[2, 2]]), (lambda x: quad_II.value (x, gamma)), (lambda x, f, tol : quad_II.grad(x, gamma)))
    df = df.append({"family": quad_II.name,"gamma": gamma, "iterations": n_iter}, ignore_index=True)
df = df.set_index("family")

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(df["gamma"].values, df["iterations"].values, label="vanilla")
ax.set_title("HCN against number of iterations for vanilla gradient descent")
ax.set_xlabel("HCN")
ax.set_ylabel("iterations")
ax.legend()

# %% [markdown]
# 3. Que constatez-vous?

# %%


# %% [markdown]
# ## Accélérations
# %% [markdown]
# Il y a des stratégies standards d'accélération de descente de gradients ; on en invente même tous les ans. Il n'y a pas nécessairement de relation d'ordre entre celles-ci, certaines sont plus adaptées que d'autres à des problèmes spécifiques et inversement. On propose d'en implémenter $3$ ici, il s'agit en particulier d'un travail bibliographique (simple).
# %% [markdown]
# 1. Implémenter la *Momentum Optimisation*. Cherchez à en comprendre le sens.

# %%


# %% [markdown]
# 2. Implémenter la *Nesterov Optimisation*. Cherchez à en comprendre le sens.

# %%


# %% [markdown]
# 3. Implémenter la *Adam Optimisation*. Cherchez à en comprendre le sens.

# %%


# %% [markdown]
# 4. Comparer les descentes de gradient pour chacune des optimisations précédentes.

# %%
df = pd.DataFrame(columns=["family", "gamma", "optimizer", "optimal_point", "optimal_value", "iterations"])
gammas = np.arange(1, 11)
gd = GD()
family_list = [quad_I, cubic_I, exp_I, multi_sinks, quad_II]
init_points = [np.array([1]), np.array([1]), np.array([1]), np.array([1]), np.array([[2, 2]])]
for family, init_point in zip(family_list, init_points):
    for gamma in gammas:

        opt_pt, iters, iters_dir, n_iter = gd(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)))
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "vanilla", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)

        opt_pt, iters, iters_dir, n_iter = gd.momentum(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)), 0.01,  0.9)
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "momentum", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)

        opt_pt, iters, iters_dir, n_iter = gd.nesterov(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)), 0.01,  0.9)
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "nesterov", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)

        opt_pt, iters, iters_dir, n_iter = gd.adam(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)), 0.01, 0.9, 0.999, 1e-8)
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "adam", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)
df = df.set_index("family")
df


# %%
nb_figures = 2*len(family_list)
fig, ax = plt.subplots(nb_figures, 1, figsize=(16, 9*nb_figures))
i = 0
optimizers = ["vanilla", "momentum", "nesterov", "adam"]
for family, init_point in zip(family_list, init_points):
    groups = df.groupby("family")
    groups = groups.get_group(family.name)
    for optimizer in optimizers:
        group = groups.loc[groups["optimizer"] == optimizer]
        ax[i].plot(group["gamma"], group["iterations"], label=optimizer)
        ax[i].set_title("optimizer comparison, gamma against iterations, family = {}".format(family.name))
        ax[i].set_xlabel("gamma")
        ax[i].set_ylabel("iterations")
        ax[i].legend()
        ax[i+1].plot(group["gamma"], group["optimal_value"], label=optimizer)
        ax[i+1].set_title("optimizer comparison, gamma against optimal value, family = {}".format(family.name))
        ax[i+1].set_xlabel("gamma")
        ax[i+1].set_ylabel("optimal value")
        ax[i+1].legend()
    i += 2

# %% [markdown]
# ## Méthode de Newton
# %% [markdown]
# 1. Implémenter la méthode de Newton.

# %%


# %% [markdown]
# 2. Qu'est-ce qu'un exemple pertinent pour comparer la méthode de Newton au méthode de descente vu précédemment?

# %%


# %% [markdown]
# 3. Comparer la méthode de Newton aux descentes précédentes.

# %%
df = pd.DataFrame(columns=["family", "gamma", "optimizer", "optimal_point", "optimal_value", "iterations"])
gammas = np.arange(1, 11)
gd = GD()
family_list = [quad_II, quad_I, cubic_I, exp_I, multi_sinks]
init_points = [np.array([[2, 2]]), np.array([1]), np.array([1]), np.array([1]), np.array([1])]
for family, init_point in zip(family_list, init_points):
    for gamma in gammas:

        opt_pt, iters, iters_dir, n_iter = gd(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)))
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "vanilla", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)

        opt_pt, iters, iters_dir, n_iter = gd.momentum(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)), 0.01,  0.9)
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "momentum", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)

        opt_pt, iters, iters_dir, n_iter = gd.nesterov(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)), 0.01,  0.9)
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "nesterov", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)

        opt_pt, iters, iters_dir, n_iter = gd.adam(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)), 0.001, 0.9, 0.999, 1e-8)
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "adam", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)

        opt_pt, iters, iters_dir, n_iter = gd.newton(init_point, (lambda x: family.value (x, gamma)), (lambda x, f, tol : family.grad(x, gamma)), (lambda x, f, tol : family.hess(x, gamma)))
        df = df.append({"family": family.name,"gamma": gamma, "optimizer": "newton", "optimal_point": opt_pt, "optimal_value": family.value(opt_pt, gamma), "iterations": n_iter}, ignore_index=True)
df = df.set_index("family")
df


# %%
nb_figures = 2*len(family_list)
fig, ax = plt.subplots(nb_figures, 1, figsize=(16, 9*nb_figures))
i = 0
optimizers = ["vanilla", "momentum", "nesterov", "adam", "newton"]
for family, init_point in zip(family_list, init_points):
    groups = df.groupby("family")
    groups = groups.get_group(family.name)
    for optimizer in optimizers:
        group = groups.loc[groups["optimizer"] == optimizer]
        ax[i].plot(group["gamma"], group["iterations"], label=optimizer)
        ax[i].set_title("optimizer comparison, gamma against iterations, family = {}".format(family.name))
        ax[i].set_xlabel("gamma")
        ax[i].set_ylabel("iterations")
        ax[i].legend()
        ax[i+1].plot(group["gamma"], group["optimal_value"], label=optimizer)
        ax[i+1].set_title("optimizer comparison, gamma against optimal value, family = {}".format(family.name))
        ax[i+1].set_xlabel("gamma")
        ax[i+1].set_ylabel("optimal value")
        ax[i+1].legend()
    i += 2

# %% [markdown]
# ## Cas pratique : la régression (YAT)
# %% [markdown]
# Dans cette section on attend de vous que vous implémtentier une régression linéaire (puis polynomiale) à la main. Il revient à vous d'effectuer l'étude bibliographique (ou de revoir les documents à votre disposition dans le cours et / ou mon git) pour éventuellement vous aider. À vous de pousser cette étude là où vous souhaitez ; j'attends de vous que vous puissiez entrevoir le besoin de régularisation, sujet abordé en second TP.

# %%
def vanilla_gd(m, c, X, Y, rate):
    i = 0
    n = float(len(X))
    Y_pred = m * X + c  # The current predicted value of Y
    D_m = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c
    while i < 1000 and (np.linalg.norm(D_m) > 1e-8 or np.linalg.norm(D_c) > 1e-8):
        Y_pred = m * X + c  # The current predicted value of Y
        D_m = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c
        m = m - rate * D_m  # Update m
        c = c - rate * D_c  # Update c
        i += 1
    return m, c

def momentum(m, c, X, Y, rate, alpha):
    i = 0
    n = float(len(X))
    Y_pred = m * X + c  # The current predicted value of Y
    delta_c = 0
    delta_m = 0
    D_m = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c
    while i < 1000 and (np.linalg.norm(D_m) > 1e-8 or np.linalg.norm(D_c) > 1e-8):
        Y_pred = m * X + c  # The current predicted value of Y
        D_m = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c
        delta_m =  alpha * delta_m - rate * D_m
        delta_c =  alpha * delta_c - rate * D_c
        m = m + delta_m # Update m
        c = c + delta_c  # Update c
        i += 1
    return m, c

def nesterov(m, c, X, Y, rate, alpha):
    i = 0
    n = float(len(X))
    Y_pred = m * X + c  # The current predicted value of Y
    D_m = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c
    delta_c = -rate * D_c
    delta_m = -rate * D_m
    while i < 1000 and (np.linalg.norm(D_m) > 1e-8 or np.linalg.norm(D_c) > 1e-8):
        Y_pred_m = (m + alpha * delta_m) * X + c  # The current predicted value of Y
        Y_pred_c = m * X + (c + alpha * delta_c)  # The current predicted value of Y
        D_m = (-2 / n) * np.sum(X * (Y - Y_pred_m))  # Derivative wrt m
        D_c = (-2 / n) * np.sum(Y - Y_pred_c)  # Derivative wrt c
        delta_m =  alpha * delta_m - rate * D_m
        delta_c =  alpha * delta_c - rate * D_c
        m = m + delta_m # Update m
        c = c + delta_c  # Update c
        i += 1
    return m, c

def adam(a, b, X, Y, rate, beta1, beta2, epsilon):
    i = 1
    m = 0
    v = 0
    n = float(len(X))
    Y_pred = a * X + b  # The current predicted value of Y
    D_a = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
    D_b = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c
    while (np.linalg.norm(D_a) > 1e-5 or np.linalg.norm(D_b) > 1e-5):

        Y_pred = a * X + b  # The current predicted value of Y
        D_a = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
        D_b = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c

        m = beta1 * m + (1 - beta1) * D_a
        mt = m / (1 - (beta1**i))
        v = beta2 * v + (1 - beta2) * (D_a**2)
        vt = v / (1 - (beta2**i))
        delta_a = -(rate * mt) / (np.sqrt(vt) + epsilon)

        m = beta1 * m + (1 - beta1) * D_b
        mt = m / (1 - (beta1**n_iter))
        v = beta2 * v + (1 - beta2) * (D_b**2)
        vt = v / (1 - (beta2**n_iter))
        delta_b = -(rate * mt) / (np.sqrt(vt) + epsilon)

        a = a + delta_a  # Update m
        b = b + delta_b  # Update c
        i += 1

    return a, b

def newton(m, c, X, Y):
    i = 0
    n = float(len(X))
    Y_pred = m * X + c  # The current predicted value of Y
    D_m = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c
    while i < 1000 and (np.linalg.norm(D_m) > 1e-8 or np.linalg.norm(D_c) > 1e-8):
        Y_pred = m * X + c  # The current predicted value of Y
        D_m = (-2 / n) * np.sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2 / n) * np.sum(Y - Y_pred)  # Derivative wrt c
        h_m = (2 / n) * np.sum(X**2)
        h_c = 2
        m = m - (1 / h_m) * D_m
        c = c - (1 / h_c) * D_c
        i += 1
    return m, c


m = 0.0
c = 0.0
m_vanilla, c_vanilla = vanilla_gd(m, c, x, y, 0.0001)
m_momentum, c_momentum = momentum(m, c, x, y, 0.0001, 0.9)
m_nesterov, c_nesterov = nesterov(m, c, x, y, 0.0001, 0.9)
m_adam, c_adam = adam(m, c, x, y, 0.001, 0.9, 0.999, 1e-8)
m_newton, c_newton = newton(m, c, x, y)
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.scatter(x, y)
ax.plot(x, m_vanilla * x + c_vanilla, label='vanilla')
ax.plot(x, m_momentum * x + c_momentum, label='momentum')
ax.plot(x, m_nesterov * x + c_nesterov, label='nesterov')
ax.plot(x, m_adam * x + c_adam, label='adam')
ax.plot(x, m_newton * x + c_newton, label='newton')
ax.legend()


# %%
def vanilla_gd(theta, X, Y, rate):
    i = 0
    n = float(X.shape[0])
    g = (1 / n) * (X.T @ ((X @ theta) - y))
    while i < 1000 and np.linalg.norm(g) > 1e-8:
        g = (1 / n) * (X.T @ ((X @ theta) - y))
        theta = theta - rate * g
        i += 1
    return theta

def momentum(theta, X, Y, rate, alpha):
    i = 0
    n = float(X.shape[0])
    g = (1 / n) * (X.T @ ((X @ theta) - y))
    delta = 0
    while i < 1000 and np.linalg.norm(g) > 1e-8:
        g = (1 / n) * (X.T @ ((X @ theta) - y))
        delta = alpha * delta - rate * g
        theta = theta + delta
        i += 1
    return theta

def nesterov(theta, X, Y, rate, alpha):
    i = 0
    n = float(X.shape[0])
    g = (1 / n) * (X.T @ ((X @ theta) - y))
    delta = 0
    while i < 1000 and np.linalg.norm(g) > 1e-8:
        g = (1 / n) * (X.T @ ((X @ (alpha * delta + theta)) - y))
        delta = alpha * delta - rate * g
        theta = theta + delta
        i += 1
    return theta

def adam(theta, X, Y, rate, beta1, beta2, epsilon):
    i = 1
    m, v = 0, 0
    n = float(X.shape[0])
    g = (1 / n) * (X.T @ ((X @ theta) - y))
    while np.linalg.norm(g) > 1e-5:
        g = (1 / n) * (X.T @ ((X @ theta) - y))

        m = beta1 * m + (1 - beta1) * g
        mt = m / (1 - (beta1**i))
        v = beta2 * v + (1 - beta2) * (g**2)
        vt = v / (1 - (beta2**i))
        delta = -(rate * mt) / (np.sqrt(vt) + epsilon)

        theta = theta + delta
        i += 1
    return theta

def newton(theta, X, Y):
    i = 0
    n = float(X.shape[0])
    g = (1 / n) * (X.T @ ((X @ theta) - y))
    while i < 1000 and np.linalg.norm(g) > 1e-8:
        g = (1 / n) * (X.T @ ((X @ theta) - y))
        h = (1 / n) * (X.T @ X)
        theta = theta - np.linalg.inv(h) @ g
        i += 1
    return theta

x = np.sort(2 - 3 * np.random.normal(0, 1, 20))
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)
x = x.reshape((-1, 1))
m = x.shape[0]
x = np.hstack((np.ones((m, 1)), x))
x = np.hstack((x, (x[:, 1] ** 2).reshape((m, 1)), (x[:, 1] ** 3).reshape((m, 1)), (x[:, 1] ** 4).reshape((m, 1))))
x[:, 1:] = (x[:, 1:] - np.mean(x[:, 1:], axis=0)) / np.std(x[:, 1:], axis=0)

theta = np.random.random(x.shape[1])

theta_vanilla = vanilla_gd(theta, x, y, 0.01)
theta_momentum = momentum(theta, x, y, 0.01, 0.9)
theta_nesterov = nesterov(theta, x, y, 0.01, 0.9)
theta_adam = adam(theta, x, y, 0.001, 0.9, 0.999, 1e-8)
theta_newton = newton(theta, x, y)
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.scatter(x[:, 1], y)
ax.plot(x[:, 1], x @ theta_vanilla, label='vanilla')
ax.plot(x[:, 1], x @ theta_momentum, label='momentum')
ax.plot(x[:, 1], x @ theta_nesterov, label='nesterov')
ax.plot(x[:, 1], x @ theta_adam, label='adam')
ax.plot(x[:, 1], x @ theta_newton, label='newton')
ax.legend()


# %%



