import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import nquad
from scipy.integrate import dblquad, tplquad, quad, trapezoid, simpson
import time
import math

# Configuration de la page
st.set_page_config(
    page_title="Analyse Numérique Intégrale",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de l'état de session
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'accueil'

# Fonctions de navigation
def navigate_to(page_name):
    st.session_state.current_page = page_name

# Sidebar - Navigation
with st.sidebar:
    st.title("Menu :")
    st.button("🏠 Accueil", on_click=navigate_to, args=('accueil',))
    st.button("➕ Intégration Numérique", on_click=navigate_to, args=('methodes_numeriques',))
    st.button("➕ Comparaison des Méthodes", on_click=navigate_to, args=('comparaison',))
    st.button("➕ Intégrales Multiples", on_click=navigate_to, args=('integrales_multiples',))
    st.button("➕ Intégral Impropre", on_click=navigate_to, args=('int_impropre',))
    st.button("➕ Intégration de Monte Carlo", on_click=navigate_to, args=('monte_carlo',))
    st.button("➕ Intégration Adaptative", on_click=navigate_to, args=('adapt',))
    st.button("📘 Guide d'utilisation", on_click=navigate_to, args=('exemples',))
    
    st.markdown("---")
    st.write("**Equipe :**")
    st.write("- BENMOUAKDEM Reda")
    st.write("- BAKHTIOUI Imad")
    st.write("- HARAMBE YAO Alpha")
    st.write("**Version:** 1.0")

# Page Accueil
if st.session_state.current_page == 'accueil':
    st.title("Analyse Numérique des Intégrales")
    st.markdown("""
    ## Bienvenue dans l'application complète d'analyse numérique des intégrales
    
    Cette application permet de :
    - Calculer des intégrales avec différentes méthodes numériques
    - Évaluer des intégrales multiples et impropres
    - Utiliser la méthode Monte Carlo
    - Comparer les différentes approches
    
    ### Fonctionnalités principales :
    """)
    
    cols = st.columns(5)
    with cols[0]:
        st.markdown("""
        **🧮 Méthodes Numériques**
        - Rectangles (gauche, droite, milieu)
        - Trapèzes
        - Simpson (1/3 et 3/8)
        - Gauss-Legendre
        """)
    
    with cols[1]:
        st.markdown("""
        **📐 Intégrales Multiples**
        - Intégrales doubles
        - Intégrales triples
        - Changement de variables
        """)
    
    with cols[2]:
        st.markdown("""
        **🎲 Monte Carlo**
        - Intégration Monte Carlo
        - Simulation stochastique
        - Analyse de convergence
        """)
    
    with cols[3]:
        st.markdown("""
        **🖇️ Intégral Impropre**
        - Intégration en bornes infini 
        """)
    with cols[4]:
        st.markdown("""
        **🔗 Intégral Adaptative**
        - Intégration avec quadrature de Simpson 
        """)
    

# Page Méthodes Numériques
elif st.session_state.current_page == 'methodes_numeriques':
    st.title("Intégration Numérique")
    st.subheader("Expression de la fonction")
    fonction = st.text_input("Entrez la fonction f(x) à intégrer", "exp(-x**2)")

    # Section pour les bornes
    st.subheader("Bornes d'intégration")
    col1, col2 = st.columns(2)
    with col1:
        a_input = st.text_input("Borne inférieure a", "-inf")
    with col2:
        b_input = st.text_input("Borne supérieure b", "inf")

    # Conversion de la fonction
    x = sp.symbols('x')
    try:
        f_expr = sp.sympify(fonction)
        f_lambda = sp.lambdify(x, f_expr, modules=['numpy', 'math'])
    except Exception as e:
        st.error(f"Erreur dans la fonction : {str(e)}")
        st.stop()

    # Fonction pour convertir les bornes
    def parse_bound(bound_str):
        bound_str = bound_str.strip().lower()
        if bound_str in ['inf', 'infini', 'infinity', '+inf', '+infini', '+infinity']:
            return sp.oo
        elif bound_str in ['-inf', '-infini', '-infinity']:
            return -sp.oo
        else:
            try:
                return sp.sympify(bound_str)
            except:
                raise ValueError(f"Impossible d'interpréter la borne: {bound_str}")

    try:
        a_val = parse_bound(a_input)
        b_val = parse_bound(b_input)
    except Exception as e:
        st.error(f"Erreur dans les bornes : {str(e)}")
        st.stop()

   # st.markdown("<br><br>", unsafe_allow_html=True)  # saut de ligne
    
    st.latex(r"\int_{" + sp.latex(a_val) + "}^{" + sp.latex(b_val) + "} " + sp.latex(f_expr) + " \, dx ")

    st.subheader("Type d'intégration: Symbolique/Numérique")
    # Sélection du type de calcul
    calculation_type = st.radio(
        "Sélectionnez le type de calcul:",
        ["Exacte (Symbolique)", "Numérique"],
        index=0
    )


    # Fonction pour convertir les bornes
    def parse_bound(bound_str):
        bound_str = bound_str.strip().lower()
        if bound_str in ['inf', 'infini', 'infinity', '+inf', '+infini', '+infinity']:
            return sp.oo
        elif bound_str in ['-inf', '-infini', '-infinity']:
            return -sp.oo
        else:
            try:
                return sp.sympify(bound_str)
            except:
                raise ValueError(f"Impossible d'interpréter la borne: {bound_str}")

    try:
        a_val = parse_bound(a_input)
        b_val = parse_bound(b_input)
    except Exception as e:
        st.error(f"Erreur dans les bornes : {str(e)}")
        st.stop()

    # Calcul Exact Symbolique
    if calculation_type == "Exacte (Symbolique)":
        try:
            integral = sp.integrate(f_expr, (x, a_val, b_val))
            
            st.subheader("Résultat")
            st.write("Fonction à intégrer:")
            # Version qui protège contre les expressions vides
            func_latex = sp.latex(f_expr) if f_expr else ""
            st.latex(rf"f(x) = {func_latex}")


            st.write("Calcul de l'intégrale:")
            st.latex(r"\int_{" + sp.latex(a_val) + "}^{" + sp.latex(b_val) + "} " + 
                sp.latex(f_expr) + " \, dx = " + sp.latex(integral))
            
            st.success(f"Résultat exact: {sp.pretty(integral)}")
            
            # Visualisation pour les bornes finies
            try:
                a_plot = float(a_val.evalf()) if a_val.is_finite else -10
                b_plot = float(b_val.evalf()) if b_val.is_finite else 10
                x_vals = np.linspace(a_plot, b_plot, 1000)
                y_vals = [f_lambda(x) for x in x_vals]
                
                # Calcul des valeurs de la primitive
                primitive = sp.integrate(f_expr, x)
                F_lambda = sp.lambdify(x, primitive, modules=['numpy', 'math'])
                F_vals = [F_lambda(x) for x in x_vals]
                
                # Création des graphiques
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Graphique de la fonction originale
                ax1.plot(x_vals, y_vals, label=f"f(x) = {fonction}", color='blue')
                if a_val.is_finite and b_val.is_finite:
                    ax1.fill_between(x_vals, y_vals, alpha=0.2, color='blue')
                ax1.set_xlabel('x')
                ax1.set_ylabel('f(x)')
                ax1.set_title("Fonction originale")
                ax1.legend()
                ax1.grid()
                
                # Graphique de la primitive
                ax2.plot(x_vals, F_vals, label=f"F(x) = {sp.latex(primitive)}", color='red')
                ax2.set_xlabel('x')
                ax2.set_ylabel('F(x)')
                ax2.set_title("Primitive (fonction intégrale)")
                ax2.legend()
                ax2.grid()
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Visualisation limitée pour cette fonction. Erreur : {str(e)}")
                
        except Exception as e:
            st.error(f"Erreur dans le calcul symbolique : {str(e)}")

    # Calcul Numérique
    elif calculation_type == "Numérique":
        has_infinite_bounds = not (a_val.is_finite and b_val.is_finite)
    
        if has_infinite_bounds:
            st.info("Les bornes infinies détectées - Utilisation automatique de la quadrature adaptative")
            decimales = st.slider("Nombre de décimales à afficher :", 2, 50, 6)
            if st.button("Calculer avec Quadrature Adaptative"):
                
                try:
                    start_time = time.time()
                    # Conversion des bornes pour scipy.quad
                    a_quad = -np.inf if a_val == -sp.oo else float(a_val)
                    b_quad = np.inf if b_val == sp.oo else float(b_val)
                    
                    result, error = integrate.quad(f_lambda, a_quad, b_quad)
                    computation_time = time.time() - start_time

                    st.success("Calcul terminé !")
                    if decimales <= 10:
                        cols = st.columns(3)
                        cols[0].metric("Résultat", f"{result:.{decimales}f}")
                        cols[1].metric("Erreur estimée", f"{error:.{decimales}f}")
                        cols[2].metric("Temps (s)", f"{computation_time:.4f}")
                    else:
                        st.write("**Résultat:**")
                        st.code(f"{result:.{decimales}f}")
                        st.write("**Erreur estimée:**")
                        st.code(f"{error:.{decimales}f}")
                        st.write("**Temps (s):**")
                        st.code(f"{computation_time:.10f}")  # On garde 10 décimales max pour le temps
                    
                    # Visualisation avec plage raisonnable
                    x_min = -10 if a_val == -sp.oo else float(a_val)
                    x_max = 10 if b_val == sp.oo else float(b_val)
                    x_vals = np.linspace(x_min, x_max, 1000)
                    y_vals = f_lambda(x_vals)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(x_vals, y_vals, label=f"f(x) = {fonction}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.legend()
                    ax.grid()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Erreur dans le calcul : {str(e)}")
        else:
            st.subheader("Paramètres numériques")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                methode = st.selectbox("Méthode numérique:", [
                "Rectangle gauche",
                "Rectangle droit",
                "Rectangle milieu",
                "Trapèzes",
                "Simpson 1/3",
                "Simpson 3/8",
                "Gauss-Legendre"
            ])
            with col2:
                n = st.slider("Nombre de subdivisions:", 10, 1000, 100)
            with col3:
                decimales = st.slider("Précision (décimales)", 2, 50, 6)
            # Sélection de la méthode
            
            
            # Nombre de subdivisions
            
            
            
            if st.button("Calculer numériquement"):
                try:
                    start_time = time.time()
                
                    a_float = float(a_val)
                    b_float = float(b_val)
                    
                    # Implémentation des méthodes avec calcul d'erreur
                    def rectangle_gauche(f, a, b, n):
                        h = (b - a)/n
                        integral = h * sum(f(a + i*h) for i in range(n))
                        # Estimation d'erreur: différence avec rectangle droit
                        integral_right = h * sum(f(a + i*h) for i in range(1, n+1))
                        error = abs(integral - integral_right)
                        return integral, error
                    
                    def rectangle_droit(f, a, b, n):
                        h = (b - a)/n
                        integral = h * sum(f(a + i*h) for i in range(1, n+1))
                        # Estimation d'erreur: différence avec rectangle gauche
                        integral_left = h * sum(f(a + i*h) for i in range(n))
                        error = abs(integral - integral_left)
                        return integral, error
                    
                    def rectangle_milieu(f, a, b, n):
                        h = (b - a)/n
                        integral = h * sum(f(a + (i+0.5)*h) for i in range(n))
                        # Estimation d'erreur: différence avec trapèzes
                        integral_trap = (h/2) * (f(a) + f(b) + 2*sum(f(a + i*h) for i in range(1, n)))
                        error = abs(integral - integral_trap)
                        return integral, error
                    
                    def trapezes(f, a, b, n):
                        h = (b - a)/n
                        integral = h/2 * (f(a) + f(b) + 2*sum(f(a + i*h) for i in range(1, n)))
                        # Estimation d'erreur: différence avec Simpson
                        if n >= 2:
                            integral_simp = (h/3) * (f(a) + f(b) + 4*sum(f(a + i*h) for i in range(1, n, 2)) + 2*sum(f(a + i*h) for i in range(2, n-1, 2)))
                            error = abs(integral - integral_simp)
                        else:
                            error = float('nan')
                        return integral, error
                    
                    def simpson_13(f, a, b, n):
                        if n % 2 != 0:
                            n += 1
                        h = (b - a)/n
                        integral = h/3 * (f(a) + f(b) + 4*sum(f(a + i*h) for i in range(1, n, 2)) + 2*sum(f(a + i*h) for i in range(2, n-1, 2)))
                        # Estimation d'erreur: différence avec n/2 points
                        if n >= 4:
                            integral_half = simpson_13(f, a, b, n//2)[0]
                            error = abs(integral - integral_half)/15
                        else:
                            error = float('nan')
                        return integral, error
                    
                    def simpson_38(f, a, b, n):
                        if n % 3 != 0:
                            n += (3 - n % 3)
                        h = (b - a)/n
                        integral = 3*h/8 * (f(a) + f(b) + 3*sum(f(a + i*h) + f(a + (i+1)*h) for i in range(1, n, 3)) + 2*sum(f(a + i*h) for i in range(3, n-2, 3)))
                        # Estimation d'erreur: différence avec n/3 points
                        if n >= 6:
                            integral_third = simpson_38(f, a, b, n//3)[0]
                            error = abs(integral - integral_third)
                        else:
                            error = float('nan')
                        return integral, error
                    
                    def gauss_legendre(f, a, b, n):
                        x, w = np.polynomial.legendre.leggauss(n)
                        t = 0.5*(b - a)*x + 0.5*(b + a)
                        integral = 0.5*(b - a) * np.sum(w * f(t))
                        # Estimation d'erreur: différence avec n-1 points
                        if n > 1:
                            x2, w2 = np.polynomial.legendre.leggauss(n-1)
                            t2 = 0.5*(b - a)*x2 + 0.5*(b + a)
                            integral2 = 0.5*(b - a) * np.sum(w2 * f(t2))
                            error = abs(integral - integral2)
                        else:
                            error = float('nan')
                        return integral, error
            
                    method_functions = {
                        "Rectangle gauche": rectangle_gauche,
                        "Rectangle droit": rectangle_droit,
                        "Rectangle milieu": rectangle_milieu,
                        "Trapèzes": trapezes,
                        "Simpson 1/3": simpson_13,
                        "Simpson 3/8": simpson_38,
                        "Gauss-Legendre": gauss_legendre
                    }
                
                    result, error = method_functions[methode](f_lambda, a_float, b_float, n)
                    computation_time = time.time() - start_time
                    st.success("Calcul terminé !")
                    if decimales <= 10:
                        cols = st.columns(3)
                        cols[0].metric("Résultat", f"{result:.{decimales}f}")
                        cols[1].metric("Erreur estimée", f"{error:.{decimales}f}")
                        cols[2].metric("Temps (s)", f"{computation_time:.4f}")
                    else:
                        st.write("**Résultat:**")
                        st.code(f"{result:.{decimales}f}")
                        st.write("**Erreur estimée:**")
                        st.code(f"{error:.{decimales}f}")
                        st.write("**Temps (s):**")
                        st.code(f"{computation_time:.10f}")  # On garde 10 décimales max pour le temps
                    
                    # Visualisation
                    x_vals = np.linspace(a_float, b_float, 1000)
                    y_vals = f_lambda(x_vals)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(x_vals, y_vals, label=f"f(x) = {fonction}")
                    
                    # On ne remplit que si les bornes sont finies
                    if a_val.is_finite and b_val.is_finite:
                        # Visualisation spécifique à la méthode
                        if methode.startswith("Rectangle"):
                            h = (b_float - a_float)/n
                            if "gauche" in methode:
                                x_rect = np.linspace(a_float, b_float-h, n)
                            elif "droit" in methode:
                                x_rect = np.linspace(a_float+h, b_float, n)
                            else:  # milieu
                                x_rect = np.linspace(a_float+h/2, b_float-h/2, n)
                            y_rect = f_lambda(x_rect)
                            ax.bar(x_rect, y_rect, width=h, alpha=0.3, align='edge' if "droit" in methode else ('center' if "milieu" in methode else 'edge'), color='orange')
                        
                        elif methode == "Trapèzes":
                            x_trap = np.linspace(a_float, b_float, n+1)
                            y_trap = f_lambda(x_trap)
                            for i in range(n):
                                ax.fill_between([x_trap[i], x_trap[i+1]], [y_trap[i], y_trap[i+1]], alpha=0.3, color='green')
                    
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.legend()
                    ax.grid()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Erreur dans le calcul : {str(e)}")
    else:
        st.info("Veuillez sélectionner un type de calcul (Exacte ou Numérique)")

# Page Comparaison
elif st.session_state.current_page == 'comparaison':
    st.title("Comparaison des Méthodes")
    st.header("Paramètres d'Entrée")
    with st.expander("Paramètres d'entrée", expanded=True):
        fonction = st.text_input("Fonction f(x)", "sin(x)")
        col1, col2, col3 = st.columns(3)
        with col1:
            a_input = st.text_input("Borne inférieure a :", value="0")
            n = st.slider("Nombre de points n :", 2, 1000, 100)
            decimales = st.slider("Nombre de décimales à afficher :", 2, 20, 6)
        with col2:
            b_input = st.text_input("Borne supérieure b :", value="pi")
            n_max = st.slider("Nombre max de points pour convergence:", 10, 5000, 1000) 
            exact_value = st.number_input("Valeur exacte (optionnelle) :", value=None)     
    
    # Bornes avec support des expressions mathématiques
    
    
    
    # Conversion des bornes en valeurs numériques
    try:
        a_expr = sp.sympify(a_input)
        a = float(a_expr.evalf())
    except:
        st.error("Expression invalide pour la borne inférieure a. Utilisez un nombre ou une expression comme '0', 'pi/2', etc.")
        st.stop()
    
    try:
        b_expr = sp.sympify(b_input)
        b = float(b_expr.evalf())
    except:
        st.error("Expression invalide pour la borne supérieure b. Utilisez un nombre ou une expression comme 'pi', 'pi/2', etc.")
        st.stop()
    
    # Vérification que a < b
    if a >= b:
        st.error("La borne inférieure a doit être strictement inférieure à la borne supérieure b")
        st.stop()
    
    
    
    

    # Conversion de la fonction en expression sympy
    x = sp.symbols('x')
    try:
        f_expr = sp.sympify(fonction)
        f_lambda = sp.lambdify(x, f_expr, modules=['numpy', {'pi': math.pi}])
    except:
        st.error("Erreur dans la syntaxe de la fonction. Utilisez 'x' comme variable et les opérations standard.")
        st.stop()

    # Calcul de la valeur exacte si non fournie
    if exact_value is None:
        try:
            exact_value = float(sp.integrate(f_expr, (x, a, b)).evalf())
        except:
            st.warning("Impossible de calculer la valeur exacte symboliquement. Les erreurs seront relatives aux autres méthodes.")
            exact_value = None

    # Fonctions pour les différentes méthodes d'intégration
    def rectangle_gauche(f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n, endpoint=False)
        return h * np.sum(f(x))

    def rectangle_droit(f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a + h, b, n)
        return h * np.sum(f(x))

    def rectangle_milieu(f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a + h/2, b - h/2, n)
        return h * np.sum(f(x))

    def trapeze(f, a, b, n):
        x = np.linspace(a, b, n)
        y = f(x)
        return (b - a) / (2 * (n - 1)) * (y[0] + y[-1] + 2 * np.sum(y[1:-1]))

    def simpson_13(f, a, b, n):
        if n % 2 != 0:
            n += 1  # Simpson nécessite un nombre pair d'intervalles
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = f(x)
        return h/3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

    def simpson_38(f, a, b, n):
        if n % 3 != 0:
            n = n + (3 - n % 3)  # Ajuster pour multiple de 3
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = f(x)
        return 3*h/8 * (y[0] + y[-1] + 3 * np.sum(y[1:-1:3] + y[2:-1:3]) + 2 * np.sum(y[3:-1:3]))

    def gauss_legendre(f, a, b, n):
        x, w = np.polynomial.legendre.leggauss(n)
        # Transformation de [-1,1] à [a,b]
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        return 0.5 * (b - a) * np.sum(w * f(t))

    def monte_carlo(f, a, b, n):
        x_rand = np.random.uniform(a, b, n)
        return (b - a) * np.mean(f(x_rand))

    # Fonction pour calculer l'erreur
    def compute_error(approx, exact):
        if exact is not None and exact != 0:
            return abs((approx - exact) / exact) * 100
        elif exact is not None:
            return abs(approx - exact)
        else:
            return None

    # Fonction pour formater les nombres avec le bon nombre de décimales
    def format_number(num, decimals):
        if num is None:
            return "N/A"
        if isinstance(num, str):
            return num
        return f"{num:.{decimals}f}"

    # Fonction pour étudier la convergence
    def study_convergence(method, f, a, b, n_max, exact):
        n_values = np.logspace(1, np.log10(n_max), 20).astype(int)
        errors = []
        for n in n_values:
            approx = method(f, a, b, n)
            err = compute_error(approx, exact)
            if err is not None:
                errors.append(err)
            else:
                errors.append(None)
        return n_values, errors

    # Bouton de calcul
    if st.button("Calculer l'Intégrale"):
        st.subheader("Résultats")
        
        # Calcul des approximations
        methods = {
            "Rectangle gauche": rectangle_gauche,
            "Rectangle droit": rectangle_droit,
            "Rectangle milieu": rectangle_milieu,
            "Trapèzes": trapeze,
            "Simpson 1/3": simpson_13,
            "Simpson 3/8": simpson_38,
            "Gauss-Legendre": gauss_legendre,
            "Monte Carlo": monte_carlo
        }
        
        results = []
        for name, method in methods.items():
            start_time = time.time()
            approx = method(f_lambda, a, b, n)
            computation_time = time.time() - start_time
            error = compute_error(approx, exact_value)
            
            # Formatage des nombres avec le bon nombre de décimales
            approx_fmt = format_number(approx, decimales)
            error_fmt = format_number(error, decimales)
            time_fmt = format_number(computation_time, 6)  # On garde 6 décimales pour le temps
            
            results.append({
                "Méthode": name,
                "Approximation": approx_fmt,
                "Erreur (%)": error_fmt,
                "Temps (s)": time_fmt
            })
        
        # Affichage des résultats sous forme de tableau
        st.table(results)
        
        # Affichage de la valeur exacte si disponible
        if exact_value is not None:
            exact_fmt = format_number(exact_value, decimales)
            st.write(f"Valeur exacte (calcul symbolique): {exact_fmt}")
        
        # Étude de convergence
        if exact_value is not None and n_max > n:
            st.subheader("Analyse de Convergence")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for name, method in methods.items():
                if name == "Monte Carlo":
                    continue  # On traite Monte Carlo séparément car aléatoire
                
                n_vals, errs = study_convergence(method, f_lambda, a, b, n_max, exact_value)
                ax.loglog(n_vals, errs, 'o-', label=name)
            
            ax.set_xlabel('Nombre de points (n)')
            ax.set_ylabel('Erreur relative (%)')
            ax.set_title('Convergence des Méthodes Numériques')
            ax.legend()
            ax.grid(True, which="both", ls="--")
            st.pyplot(fig)
            
            # Convergence Monte Carlo (moyenne sur plusieurs essais)
            st.write("Convergence de la méthode Monte Carlo (moyenne sur 10 essais):")
            n_vals_mc = np.logspace(1, np.log10(n_max), 10).astype(int)
            avg_errs_mc = []
            
            for n in n_vals_mc:
                errs = []
                for _ in range(10):
                    approx = monte_carlo(f_lambda, a, b, n)
                    err = compute_error(approx, exact_value)
                    errs.append(err)
                avg_errs_mc.append(np.mean(errs))
            
            fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
            ax_mc.loglog(n_vals_mc, avg_errs_mc, 'o-', label="Monte Carlo")
            ax_mc.set_xlabel('Nombre de points (n)')
            ax_mc.set_ylabel('Erreur relative moyenne (%)')
            ax_mc.set_title('Convergence de la Méthode Monte Carlo')
            ax_mc.legend()
            ax_mc.grid(True, which="both", ls="--")
            st.pyplot(fig_mc)
        
        # Théorie sur les méthodes
        st.subheader("Informations sur les Méthodes")
        st.write("""
        | Méthode           | Ordre de Convergence | Erreur Théorique | Commentaires |
        |-------------------|----------------------|------------------|--------------|
        | Rectangle gauche  | O(h)                 | O((b-a)²/n)      | Méthode d'ordre 1 |
        | Rectangle droit   | O(h)                 | O((b-a)²/n)      | Méthode d'ordre 1 |
        | Rectangle milieu  | O(h²)                | O((b-a)³/n²)     | Méthode d'ordre 2 |
        | Trapèzes          | O(h²)                | O((b-a)³/n²)     | Méthode d'ordre 2 |
        | Simpson 1/3       | O(h⁴)                | O((b-a)⁵/n⁴)     | Méthode d'ordre 4 (nécessite n pair) |
        | Simpson 3/8       | O(h⁴)                | O((b-a)⁵/n⁴)     | Méthode d'ordre 4 (nécessite n multiple de 3) |
        | Gauss-Legendre    | O(h²ⁿ)               | Très rapide      | Méthode spectrale, exacte pour polynômes de degré ≤ 2n-1 |
        | Monte Carlo       | O(1/√n)              | Non déterministe | Convergence lente mais efficace en haute dimension |
        """)
        
        # Création du graphique combiné
        st.subheader("Visualisation de la Fonction et son Intégrale")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Courbe de la fonction originale (axe y de gauche)
        x_vals = np.linspace(a, b, 500)
        y_vals = f_lambda(x_vals)
        ax1.plot(x_vals, y_vals, 'b-', label=f"f(x) = {fonction}")
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        # Calcul numérique de la primitive F(x) = ∫f(t)dt de a à x
        F_vals = np.zeros_like(x_vals)
        for i in range(1, len(x_vals)):
            F_vals[i] = np.trapz(y_vals[:i+1], x=x_vals[:i+1])

        # Création d'un second axe y pour la primitive
        ax2 = ax1.twinx()
        ax2.plot(x_vals, F_vals, 'r-', label=f"F(x) = ∫f(t)dt de {a_input} à x")
        ax2.set_ylabel('F(x)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Ajout des légendes combinées
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Affichage
        plt.title(f"Fonction et son Intégrale entre {a_input} et {b_input}")
        st.pyplot(fig)

        # Affichage des valeurs
        st.write(f"Valeur en {b_input}:")
        st.write(f"- f({b_input}) = {y_vals[-1]:.{decimales}f}")
        if exact_value is not None:
            st.write(f"- F({b_input}) = {exact_value:.{decimales}f} (exact)")
        else:
            st.write(f"- F({b_input}) ≈ {F_vals[-1]:.{decimales}f} (numérique)")

# Page Intégrales Multiples
elif st.session_state.current_page == 'integrales_multiples':
    st.title("Intégrale Multiple")
    st.subheader("Paramètres d'Entrée")
    # Entrée de l'expression
    expr_input = st.text_input("Entrez l'expression à intégrer :", "x*y**2")

    # Nombre de variables
    num_vars = st.selectbox("Nombre de variables :", [2, 3, 4], index=0)

    # Symboles possibles
    symbols = sp.symbols("x y z w")
    variables = [symbols[i] for i in range(num_vars)]

    st.subheader("Mode de Calcul")

    # Choix du type de calcul
    calculation_mode = st.radio("Choisier le mode de calcul :", 
                            ["Symbolique sans bornes", 
                                "Calcul avec bornes (exact/numérique)"],
                            index=0)

    if calculation_mode == "Symbolique sans bornes":
        # Calcul symbolique simple sans bornes
        try:
            expr = sp.sympify(expr_input)
            start_time = time.time()
            integral = sp.integrate(expr, *variables)
            computation_time = time.time() - start_time
            
            st.markdown("### 📘 Forme de l'intégrale :")
            st.latex(sp.latex(sp.Integral(expr, *variables)))
            
            st.subheader("Résultat Symbolique")
            st.latex(sp.latex(integral))
            st.success(f"Primitive : {sp.pretty(integral)}")
            st.info(f"Temps de calcul : {computation_time:.4f} secondes")
            
        except Exception as e:
            st.error(f"Erreur dans le calcul symbolique : {str(e)}")

    else:
        # Mode avec bornes
        integration_info = []
        bounds = []
        has_bounds = True
        
        # Définir les variables et les bornes
        for i in range(num_vars):
            var = variables[i]
            col1, col2 = st.columns(2)
            with col1:
                lower = st.text_input(f"Borne inférieure pour {var}", key=f"lower_{var}")
            with col2:
                upper = st.text_input(f"Borne supérieure pour {var}", key=f"upper_{var}")
            
            # Vérification des bornes
            if not lower or not upper:
                has_bounds = False
                break
            
            try:
                a = sp.sympify(lower)
                b = sp.sympify(upper)
                integration_info.append((var, a, b))
                bounds.append((float(a.evalf()), float(b.evalf())))
            except:
                st.error(f"Erreur dans les bornes de {var}")
                st.stop()
        
        if not has_bounds:
            st.warning("Veuillez spécifier toutes les bornes pour le calcul")
            st.stop()
        
        try:
            f_expr = sp.sympify(expr_input)
            integrale_latex = sp.latex(
                sp.Integral(f_expr, *[(var, a, b) for var, a, b in integration_info])
            )
            st.markdown("### 📘 Forme de l'intégrale :")
            st.latex(integrale_latex)
        except Exception as e:
            st.error(f"Erreur dans l'expression ou les bornes : {e}")
            st.stop()

        # Choix du type de calcul (exact ou numérique)
        calculation_type = st.radio("Type de calcul :", ["Exact (symbolique)", "Numérique"], index=0)
        
        # Options communes
        simplify = st.checkbox("Simplifier le résultat ?", value=True)

        if calculation_type == "Exact (symbolique)":
            # Calcul symbolique avec bornes
            try:
                expr = sp.sympify(expr_input)
                start_time = time.time()
                integral = sp.integrate(expr, *integration_info)
                computation_time = time.time() - start_time
                
                if simplify:
                    integral = sp.simplify(integral)
                
                st.subheader("Résultat Exact")
                st.latex(sp.latex(integral))
                st.success(f"Valeur exacte : {sp.pretty(integral)}")
                st.info(f"Temps de calcul : {computation_time:.4f} secondes")

                
                # Visualisation de la fonction (pour 1D ou 2D)
                st.subheader("Visualisation de la fonction")
                
                if num_vars == 1:
                    try:
                        # Cas 1D
                        x_var = variables[0]
                        a, b = bounds[0]
                        x_vals = np.linspace(float(a), float(b), 500)
                        f_lambda = sp.lambdify(x_var, expr, "numpy")
                        y_vals = f_lambda(x_vals)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(x_vals, y_vals, label=f"f({x_var}) = {expr_input}")
                        ax.fill_between(x_vals, y_vals, alpha=0.3)
                        ax.set_xlabel(str(x_var))
                        ax.set_ylabel(f"f({x_var})")
                        ax.legend()
                        ax.grid()
                        st.pyplot(fig)
                        
                    except:
                        st.warning("Visualisation 1D non disponible pour cette fonction")

                elif num_vars == 2:
                    try:
                        # Cas 2D
                        x_var, y_var = variables[:2]
                        x_bounds, y_bounds = bounds[:2]
                        
                        x_vals = np.linspace(float(x_bounds[0]), float(x_bounds[1]), 100)
                        y_vals = np.linspace(float(y_bounds[0]), float(y_bounds[1]), 100)
                        X, Y = np.meshgrid(x_vals, y_vals)
                        
                        f_lambda = sp.lambdify((x_var, y_var), expr, "numpy")
                        Z = f_lambda(X, Y)
                        
                        fig = plt.figure(figsize=(12, 5))
                        
                        # Surface plot
                        ax1 = fig.add_subplot(121, projection='3d')
                        ax1.plot_surface(X, Y, Z, cmap='viridis')
                        ax1.set_xlabel(str(x_var))
                        ax1.set_ylabel(str(y_var))
                        ax1.set_title(f"f({x_var},{y_var})")
                        
                        # Contour plot
                        ax2 = fig.add_subplot(122)
                        cs = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
                        plt.colorbar(cs)
                        ax2.set_xlabel(str(x_var))
                        ax2.set_ylabel(str(y_var))
                        ax2.set_title("Valeurs de la fonction")
                        
                        st.pyplot(fig)
                        
                    except:
                        st.warning("Visualisation 2D non disponible pour cette fonction")

                else:
                    st.info("Visualisation disponible seulement pour 1 ou 2 variables")

            except Exception as e:
                st.error(f"Erreur dans le calcul exact : {str(e)}")

        else:
            # Calcul numérique
            st.subheader("Options Numériques")
            method = st.selectbox("Méthode numérique :", 
                                ["Quadrature adaptative", "Monte Carlo", "Quadrature de Gauss"])
            
            if method == "Quadrature adaptative":
                n_points = st.slider("Précision (points d'évaluation)", 100, 10000, 1000)
            elif method == "Monte Carlo":
                n_points = st.slider("Nombre de points aléatoires", 1000, 1000000, 100000)
            else:
                n_points = st.slider("Ordre de quadrature", 5, 50, 20)

            if st.button("Calculer numériquement"):
                try:
                    expr = sp.sympify(expr_input)
                    f = sp.lambdify(variables, expr, "numpy")
                    
                    start_time = time.time()
                    
                    if method == "Quadrature adaptative":
                        result, error = nquad(f, bounds, 
                                            opts={'limit': n_points//(10*num_vars)})
                        
                    elif method == "Monte Carlo":
                        # Implémentation Monte Carlo
                        def monte_carlo(f, bounds, n):
                            dim = len(bounds)
                            points = np.random.uniform(size=(n, dim))
                            for i in range(dim):
                                a, b = bounds[i]
                                points[:,i] = points[:,i]*(b-a) + a
                            volume = np.prod([b-a for a,b in bounds])
                            return volume * np.mean(f(*points.T))
                        
                        result = monte_carlo(f, bounds, n_points)
                        error = np.nan
                        
                    else:  # Quadrature de Gauss
                        from numpy.polynomial.legendre import leggauss
                        
                        def gauss_quad(f, bounds, n):
                            points, weights = [], []
                            for a, b in bounds:
                                xi, wi = leggauss(n)
                                xi = 0.5*(xi + 1)*(b - a) + a
                                wi = 0.5*(b - a)*wi
                                points.append(xi)
                                weights.append(wi)
                            
                            grid = np.meshgrid(*points, indexing='ij')
                            W = np.ones_like(grid[0])
                            for w in weights:
                                W = W * np.resize(w, W.shape)
                            
                            return np.sum(f(*grid) * W)
                        
                        result = gauss_quad(f, bounds, n_points)
                        error = np.nan
                    
                    computation_time = time.time() - start_time
                    
                    # Affichage des résultats
                    st.subheader("Résultat Numérique")
                    cols = st.columns(3)
                    cols[0].metric("Valeur", f"{result:.8f}")
                    cols[1].metric("Erreur", f"{error:.2e}" if not np.isnan(error) else "N/A")
                    cols[2].metric("Temps", f"{computation_time:.4f}s")
                    
                    st.subheader("Visualisation graphique :")
                    # Visualisation pour 2D
                    if num_vars == 2:
                        try:
                            x = np.linspace(bounds[0][0], bounds[0][1], 100)
                            y = np.linspace(bounds[1][0], bounds[1][1], 100)
                            X, Y = np.meshgrid(x, y)
                            Z = f(X, Y)
                            
                            fig = plt.figure(figsize=(12, 5))
                            ax1 = fig.add_subplot(121, projection='3d')
                            ax1.plot_surface(X, Y, Z, cmap='viridis')
                            ax1.set_title("Fonction à intégrer")
                            
                            ax2 = fig.add_subplot(122)
                            cs = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
                            fig.colorbar(cs)
                            ax2.set_title("Valeurs de la fonction")
                            
                            st.pyplot(fig)
                        except:
                            st.warning("Visualisation 3D non disponible pour cette fonction")

                except Exception as e:
                    st.error(f"Erreur dans le calcul numérique : {str(e)}")

# Page Intégral impropre
elif st.session_state.current_page == 'int_impropre':
    st.title("Méthodes de Calcul des Intégrales Impropres")
    
    st.subheader("Paramètres d'Entrée")
    with st.expander("Paramètres de Calcul", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fonction = st.text_input("Fonction f(x) :", "exp(-x**2)")
            a = st.text_input("Borne inférieure a :", "-inf")
            decimales = st.slider("Précision (décimales) :", 2, 15, 6)
            
        with col2:
            methode = st.selectbox("Méthode :", ["Quadrature adaptative", "Changement de variable"])
            b = st.text_input("Borne supérieure b :", "inf")
            max_points = st.number_input("Points max d'évaluation :", 100, 100000, 10000)
    
    # Fonction pour convertir les bornes en valeurs numériques
    def parse_bound(bound_str):
        bound_str = bound_str.strip().lower()
        if bound_str in ['inf', 'infty', 'infini', '+inf', '+infty', '+infini']:
            return np.inf
        elif bound_str in ['-inf', '-infty', '-infini']:
            return -np.inf
        else:
            try:
                return float(sp.sympify(bound_str).evalf())
            except:
                raise ValueError(f"Impossible d'interpréter la borne: {bound_str}")

    # Conversion des bornes et de la fonction
    x = sp.symbols('x')
    try:
        f_expr = sp.sympify(fonction)
        f_lambda = sp.lambdify(x, f_expr, modules=['numpy', 'math'])
        
        a_val = parse_bound(a)
        b_val = parse_bound(b)
            
    except Exception as e:
        st.error(f"Erreur dans la saisie : {str(e)}")
        st.stop()

    # Fonctions de calcul
    def compute_adaptive(f, a, b, tol=1e-6):
        return integrate.quad(f, a, b, limit=max_points)

    if st.button("Calculer l'intégrale"):
        start_time = time.time()
        
        with st.spinner("Calcul en cours..."):
            try:
                result, error = compute_adaptive(f_lambda, a_val, b_val)
                computation_time = time.time() - start_time
                
                # Affichage des résultats
                st.success("Calcul terminé !")
                cols = st.columns(3)
                cols[0].metric("Résultat", f"{result:.{decimales}f}")
                cols[1].metric("Erreur estimée", f"{error:.{decimales}f}")
                cols[2].metric("Temps de calcul", f"{computation_time:.4f} sec")
                
                # Visualisation
                st.subheader("Visualisation de la fonction")
                try:
                    # Détermination des bornes de visualisation
                    if np.isinf(a_val) and np.isinf(b_val):
                        x_min, x_max = -5, 5
                    elif np.isinf(a_val):
                        x_min, x_max = float(b_val)-10, float(b_val)
                    elif np.isinf(b_val):
                        x_min, x_max = float(a_val), float(a_val)+10
                    else:
                        x_min, x_max = float(a_val), float(b_val)
                    
                    x_vals = np.linspace(x_min, x_max, 1000)
                    y_vals = f_lambda(x_vals)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(x_vals, y_vals, label=f"f(x) = {fonction}")
                    
                    # Remplissage seulement si les bornes sont finies
                    if not np.isinf(a_val) and not np.isinf(b_val):
                        ax.fill_between(x_vals, y_vals, alpha=0.2)
                    
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.legend()
                    ax.grid()
                    st.pyplot(fig)
                
                except Exception as e:
                    st.warning(f"Visualisation limitée : {str(e)}")

            except Exception as e:
                st.error(f"Erreur dans le calcul : {str(e)}")

# Page Monte Carlo
elif st.session_state.current_page == 'monte_carlo':
    st.title("Intégration de Monte Carlo - Multidimensionnelle")

    st.header("Paramètres d'Entrée :")
    # Entrée de l'expression
    expr_input = st.text_input("Entrez la fonction f(x, y, ...)", "x**2 + y**2")

    # Nombre de variables
    num_vars = st.selectbox("Nombre de variables :", [1, 2, 3, 4], index=1)

    # Symboles possibles
    symbols = sp.symbols("x y z w")[:num_vars]
    bounds_input = []  # Stocker les entrées utilisateur
    bounds = []       # Stocker les valeurs numériques

    # Définir les bornes pour chaque variable
    for i, var in enumerate(symbols):
        col1, col2 = st.columns(2)
        with col1:
            a_input = st.text_input(f"Borne inférieure pour {var}", "-1", key=f"low_{var}_{i}")
        with col2:
            b_input = st.text_input(f"Borne supérieure pour {var}", "1", key=f"up_{var}_{i}")
        bounds_input.append((a_input, b_input))
        try:
            a_sym = sp.sympify(a_input)
            b_sym = sp.sympify(b_input)
            bounds.append((float(a_sym.evalf()), float(b_sym.evalf())))
        except:
            st.error(f"Borne invalide pour {var}")
            st.stop()

    # Affichage de la forme de l'intégrale
    try:
        f_expr = sp.sympify(expr_input)
        # Créer les bornes symboliques pour l'affichage
        integral_bounds = [(v, sp.sympify(a), sp.sympify(b)) for v, (a, b) in zip(symbols, bounds_input)]
        integral_latex = sp.latex(sp.Integral(f_expr, *[(v, a, b) for v, a, b in integral_bounds]))
        st.markdown("### 🧾 Forme de l'intégrale :")
        st.latex(integral_latex)
    except Exception as e:
        st.error(f"Erreur dans la fonction : {e}")
        st.stop()

    # Nombre d'échantillons (avec une clé unique)
    N = st.number_input("Nombre de points Monte Carlo", 
                    min_value=1000, 
                    max_value=1_000_000, 
                    value=10000, 
                    step=1000,
                    key="num_points_input")

    # Bouton de calcul (avec une clé unique)
    if st.button("Calculer l'intégrale (Monte Carlo)", key="calculate_button"):
        try:
            f_lambda = sp.lambdify(symbols, f_expr, modules=["numpy"])

            # Générer des points aléatoires dans l'hyperrectangle
            points = np.random.rand(N, num_vars)
            scaled_points = np.zeros_like(points)
            vol = 1.0

            for i in range(num_vars):
                a, b = bounds[i]
                scaled_points[:, i] = a + (b - a) * points[:, i]
                vol *= (b - a)

            f_vals = f_lambda(*[scaled_points[:, i] for i in range(num_vars)])
            f_vals = np.nan_to_num(f_vals)

            mean_val = np.mean(f_vals)
            std_err = np.std(f_vals) / np.sqrt(N)
            estimate = vol * mean_val
            error = vol * std_err

            st.success(f"Valeur estimée de l'intégrale : {estimate:.6f}")
            st.info(f"Erreur standard estimée : ± {error:.6f}")

            # =============================================
            # VISUALISATIONS
            # =============================================
            
            st.markdown("## 📊 Visualisations")

            # 1. Nuage de points pour 2D
            if num_vars == 2 and N <= 10000:
                st.markdown("### Nuage de points 2D")
                fig, ax = plt.subplots()
                x_vals = scaled_points[:, 0]
                y_vals = scaled_points[:, 1]
                
                scatter = ax.scatter(x_vals, y_vals, c=f_vals, cmap='viridis', s=5)
                plt.colorbar(scatter, label='Valeur de f(x,y)')
                
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Échantillons Monte Carlo')
                st.pyplot(fig)

            # 2. Histogramme des valeurs
            st.markdown("### Distribution des valeurs")
            fig2, ax2 = plt.subplots()
            ax2.hist(f_vals, bins=50, density=True, alpha=0.7)
            ax2.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'Moyenne: {mean_val:.2f}')
            ax2.set_xlabel('Valeurs de f(x)')
            ax2.set_ylabel('Densité')
            ax2.set_title('Distribution des valeurs échantillonnées')
            ax2.legend()
            st.pyplot(fig2)

            # 3. Convergence de la méthode Monte Carlo (améliorée)
            st.markdown("### Convergence de la méthode Monte Carlo")
            
            # Calculer la convergence progressive
            sample_sizes = np.logspace(np.log10(100), np.log10(N), 100).astype(int)
            estimates = []
            errors = []
            
            for size in sample_sizes:
                sub_sample = f_vals[:size]
                m = np.mean(sub_sample)
                estimates.append(vol * m)
                errors.append(vol * np.std(sub_sample)/np.sqrt(size))
            
            # Créer la figure
            fig3, (ax3_1, ax3_2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Graphique supérieur: Estimation et intervalle de confiance
            ax3_1.plot(sample_sizes, estimates, 'b-', label='Estimation')
            ax3_1.fill_between(sample_sizes, 
                            np.array(estimates) - np.array(errors), 
                            np.array(estimates) + np.array(errors),
                            color='b', alpha=0.2, label='Intervalle de confiance')
            ax3_1.axhline(estimate, color='r', linestyle='--', label='Valeur finale')
            ax3_1.set_xscale('log')
            ax3_1.set_xlabel('Nombre d\'échantillons (échelle log)')
            ax3_1.set_ylabel('Valeur estimée')
            ax3_1.set_title('Convergence de l\'estimation')
            ax3_1.legend()
            ax3_1.grid(True)
            
            # Graphique inférieur: Erreur relative
            final_estimate = estimates[-1]
            relative_errors = np.abs((np.array(estimates) - final_estimate)/final_estimate)
            ax3_2.loglog(sample_sizes, relative_errors, 'g-', label='Erreur relative')
            ax3_2.set_xlabel('Nombre d\'échantillons (échelle log)')
            ax3_2.set_ylabel('Erreur relative')
            ax3_2.set_title('Convergence de l\'erreur relative')
            ax3_2.legend()
            ax3_2.grid(True)
            
            plt.tight_layout()
            st.pyplot(fig3)

            # 4. Visualisation 3D (pour 2 variables)
            if num_vars == 2:
                st.markdown("### Surface 3D de la fonction")
                fig4 = plt.figure(figsize=(10, 7))
                ax4 = fig4.add_subplot(111, projection='3d')
                
                # Créer une grille pour la surface
                x_grid = np.linspace(bounds[0][0], bounds[0][1], 50)
                y_grid = np.linspace(bounds[1][0], bounds[1][1], 50)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = f_lambda(X, Y)
                
                # Surface de la fonction
                surf = ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
                fig4.colorbar(surf, ax=ax4, shrink=0.5, aspect=5)
                
                # Points échantillonnés (un échantillon pour éviter la surcharge)
                if N <= 10000:
                    ax4.scatter(scaled_points[:, 0], scaled_points[:, 1], f_vals, 
                            c='r', s=1, alpha=0.3, label='Points MC')
                
                ax4.set_xlabel('x')
                ax4.set_ylabel('y')
                ax4.set_zlabel('f(x,y)')
                ax4.set_title('Surface de la fonction avec points échantillonnés')
                ax4.legend()
                st.pyplot(fig4)

        except Exception as e:
            st.error(f"Erreur dans le calcul Monte Carlo : {e}")

# Page Intégration adaptative
elif st.session_state.current_page == 'adapt':
    st.title("Intégration Adaptative")
    st.subheader("Paramètres d'Entrée")
    # Définition globale de la méthode de Simpson
    def simpson(f, a, b):
        c = (a + b) / 2
        h3 = (b - a) / 6
        return h3 * (f(a) + 4*f(c) + f(b))

    # Section paramètres
    
    with st.expander("Paramètres de calcul", expanded=True):
        fonction = st.text_input("Fonction f(x)", "sin(x)", 
                                help="Utilisez 'x' comme variable. Ex: x**2, exp(-x), sin(pi*x)")
        col1, col2 = st.columns(2)
        
        with col1:
            
            a = st.number_input("Borne inférieure a", -10.0, 10.0, 0.0, 
                            step=0.1, format="%.2f")
            
        with col2:
            b = st.number_input("Borne supérieure b", -10.0, 10.0, float(np.pi), 
                            step=0.1, format="%.2f")

        st.markdown("---")
        max_depth = st.slider("Profondeur max", 1, 20, 10,
                                help="Nombre maximal de subdivisions récursives")
        tol_options = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        tol = st.select_slider(
            "Précision (tolérance)",
            options=tol_options,
            value=1e-6,
            format_func=lambda x: f"{x:.1e}",
            help="Plus la valeur est petite, plus le calcul est précis mais long"
        )

    # Conversion de la fonction
    try:
        f = lambda x: eval(fonction, {'x': x, 'np': np, 'sin': np.sin, 'cos': np.cos, 
                                    'tan': np.tan, 'exp': np.exp, 'log': np.log,
                                    'sqrt': np.sqrt, 'pi': np.pi})
    except Exception as e:
        st.error(f"Erreur dans la fonction : {e}")
        st.stop()

    # Algorithme d'intégration adaptative
    def adaptive_integration(f, a, b, tol, depth=0, max_depth=10):    
        c = (a + b) / 2
        integral_ab = simpson(f, a, b)
        integral_ac = simpson(f, a, c)
        integral_cb = simpson(f, c, b)
        error = abs(integral_ab - (integral_ac + integral_cb))
        
        if error < tol or depth >= max_depth:
            return integral_ac + integral_cb
        else:
            return (adaptive_integration(f, a, c, tol/2, depth+1, max_depth) + 
                    adaptive_integration(f, c, b, tol/2, depth+1, max_depth))

    # Fonction pour tracer les subdivisions
    def plot_adaptive(f, a, b, tol, depth=0, max_depth=10):
        c = (a + b) / 2
        plt.plot([a, a], [0, f(a)], 'r-', alpha=0.3)
        plt.plot([b, b], [0, f(b)], 'r-', alpha=0.3)
        
        if depth < max_depth:
            error = abs(simpson(f, a, b) - (simpson(f, a, c) + simpson(f, c, b)))
            if error > tol:
                plot_adaptive(f, a, c, tol/2, depth+1, max_depth)
                plot_adaptive(f, c, b, tol/2, depth+1, max_depth)

    # Calcul et visualisation
    try:
        # Calcul des résultats
        result_adaptive = adaptive_integration(f, a, b, tol, 0, max_depth)
        result_scipy, _ = integrate.quad(f, a, b, epsabs=tol)
        
        # Préparation du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = np.linspace(a, b, 500)
        y_vals = f(x_vals)
        ax.plot(x_vals, y_vals, 'b-', label=f"f(x) = {fonction}")
        
        # Tracé des subdivisions
        plot_adaptive(f, a, b, tol, 0, max_depth)
        ax.fill_between(x_vals, y_vals, alpha=0.2)
        ax.set_title(f"Subdivisions adaptatives (tolérance = {tol:.1e})")
        ax.legend()
        
        # Affichage
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Méthode adaptative", f"{result_adaptive:.8f}")
        with col2:
            st.metric("Scipy (référence)", f"{result_scipy:.8f}")
        
    except Exception as e:
        st.error(f"Erreur de calcul : {e}")

# Page Guide
elif st.session_state.current_page == 'exemples':
    st.title("📘 Guide d'Utilisation")

    st.markdown("""
    Cette application permet de **calculer des intégrales** en utilisant différentes méthodes numériques 
    et affiche des **visualisations interactives** de la fonction et de son intégrale.
    """)

    with st.expander("📌 1. Paramètres d'Entrée", expanded=True):
        st.subheader("Fonction à intégrer (f(x))")
        st.markdown("""
        - **Syntaxe** : Utilisez `x` comme variable et des expressions mathématiques valides.
        - **Exemples** :
            - `sin(x)` → $\sin(x)$
            - `cos(x)` → $\cos(x)$
            - `tan(x)` → $\\tan(x)$
            - `arcsin(x)` → $\\arcsin(x)$
            - `arcscos(x)` → $\\arccos(x)$
            - `arctan(x)` → $\\arctan(x)$
            - `exp(x)` → $e^x$
            - `x**2` ou `x^2` → $x^2$
            - `x**n` ou `x^n` → $x^n$ 
            - `x**2 + 3*x - 5` → $x^2 + 3x - 5$
            - `1/x` → $\\frac{1}{x}$
            - `1/(1+x**2)` → $\\frac{1}{1+x^2}$
            - `x^2-1/(1+x**2)` → $\\frac{x^2-1}{1+x^2}$
            - `log(x)` → $\ln(x)$ (logarithme naturel)
            - `sqrt(x)` → $\sqrt{x}$
            - `pi * cos(x)` → $\pi \cos(x)$
            - `abs(x)` → $|x|$
            - `alpha` → $\\alpha$
            - `beta` → $\\beta$
            - `gamma` → $\\gamma$
            - `delta` → $\\delta$
            - `epsilon` → $\\epsilon$
            - `zeta` → $\\zeta$
            - `eta` → $\\eta$
            - `theta` → $\\theta$
            - `iota` → $\\iota$
            - `kappa` → $\\kappa$
            - `lambda` → $\\lambda$
            - `mu` → $\\mu$
            - `nu` → $\\nu$
            - `xi` → $\\xi$
            - `pi` → $\\pi$
            - `rho` → $\\rho$
            - `sigma` → $\\sigma$
            - `tau` → $\\tau$
            - `phi` → $\\phi$
            - `chi` → $\\chi$
            - `psi` → $\\psi$
            - `omega` → $\\omega$
            - `Gamma` → $\\Gamma$
            - `Delta` → $\\Delta$
            - `Theta` → $\\Theta$
            - `Lambda` → $\\Lambda$
            - `Xi` → $\\Xi$
            - `Pi` → $\\Pi$
            - `Sigma` → $\\Sigma$
            - `Phi` → $\\Phi$
            - `Psi` → $\\Psi$
            - `Omega` → $\\Omega$
        """)

        st.subheader("Bornes d'intégration (a et b)")
        st.markdown("""
        - **Syntaxe** : Nombres ou expressions mathématiques.
        - **Exemples** :
            - `0` → $0$
            - `pi` → $\pi$
            - `pi/2` → $\\frac{\pi}{2}$
            - `+inf` → $+\infty$
            - `-inf` → $-\infty$
            - `1.5` → $1.5$
            - `sqrt(2)` → $\sqrt{2}$
        """)

        st.subheader("Nombre de points (n)")
        st.markdown("""
        - **Définition** : Nombre de subdivisions pour les méthodes numériques.
        - **Recommandé** : Entre 100 et 1000 pour un bon équilibre précision/temps.
        """)

        st.subheader("Valeur exacte (optionnelle)")
        st.markdown("""
        - **Utilité** : Permet de comparer avec les méthodes numériques.
        - **Exemple** : Si vous intégrez `sin(x)` entre `0` et `pi`, la valeur exacte est `2`.
        """)

    with st.expander("📊 2. Méthodes d'Intégration Disponibles", expanded=False):
        st.markdown("""
        | Méthode          | Précision (Ordre) | Utilisation Recommandée |
        |------------------|------------------|------------------------|
        | **Rectangle gauche** | $O(h)$ | Simple mais peu précis |
        | **Rectangle droit** | $O(h)$ | Similaire à gauche |
        | **Rectangle milieu** | $O(h^2)$ | Plus précis que gauche/droit |
        | **Trapèzes** | $O(h^2)$ | Bon compromis vitesse/précision |
        | **Simpson 1/3** | $O(h^4)$ | Très précis pour fonctions lisses |
        | **Simpson 3/8** | $O(h^4)$ | Alternative à Simpson 1/3 |
        | **Gauss-Legendre** | $O(h^{2n})$ | Extrêmement précis pour petits `n` |
        | **Monte Carlo** | $O(1/\sqrt{n})$ | Utile en haute dimension |
        """)

    with st.expander("📈 3. Visualisations Disponibles", expanded=False):
        st.markdown("""
        1. **Graphique de la fonction**  
        - Affiche $f(x)$ entre $a$ et $b$.
        - Exemple :  
            ```python
            f(x) = sin(x) → affiche la courbe sinusoïdale.
            ```

        2. **Aire sous la courbe** *(Intégrale)*  
        - Montre la zone calculée par l'intégrale.
        - **Vert** : Partie positive ($f(x) \geq 0$).
        - **Rouge** : Partie négative ($f(x) < 0$).

        3. **Convergence des méthodes**  
        - Compare l'erreur relative en fonction du nombre de points $n$.

        4. **Primitive $F(x) = \int_a^x f(t) \, dt$** *(Optionnel)*  
        - Affiche la fonction intégrale dans le même repère que $f(x)$.
        """)

    with st.expander("⚙️ 4. Paramètres Avancés", expanded=False):
        st.markdown("""
        | Option | Description |
        |--------|-------------|
        | **Nombre de décimales** | Précision d'affichage des résultats (2 à 20 chiffres). |
        | **Nombre max de points** | Limite pour l'étude de convergence (10 à 5000). |
        """)

    with st.expander("🚀 5. Exemples d'Utilisation", expanded=False):
        st.markdown("""
        1. **Intégrale de $\sin(x)$ entre $0$ et $\pi$**
        - Fonction : `sin(x)`
        - Bornes : `a = 0`, `b = pi`
        - Valeur exacte : `2`

        2. **Intégrale de $e^{-x^2}$ entre $0$ et $1$**
        - Fonction : `exp(-x**2)`
        - Bornes : `a = 0`, `b = 1`
        - Valeur exacte : (non fournie → calcul numérique)

        3. **Intégrale de $\frac{1}{1+x^2}$ entre $-1$ et $1$**
        - Fonction : `1/(1+x**2)`
        - Bornes : `a = -1`, `b = 1`
        - Valeur exacte : `pi/2`
        """)

    with st.expander("❓ 6. FAQ", expanded=False):
        st.markdown("""
        ### **Q1. Pourquoi certaines méthodes donnent-elles des erreurs ?**
        - Les méthodes d'ordre bas (Rectangle) sont moins précises pour des fonctions complexes.
        - **Solution** : Utilisez **Simpson** ou **Gauss-Legendre** pour plus de précision.

        ### **Q2. Comment améliorer la précision ?**
        - Augmentez le nombre de points (`n`).
        - Utilisez des méthodes d'ordre élevé (**Simpson 3/8**, **Gauss-Legendre**).

        ### **Q3. Pourquoi Monte Carlo est-il lent ?**
        - Méthode aléatoire → nécessite beaucoup de points pour converger.
        - **Utile en haute dimension** (pas ici, car 1D).
        """)

    with st.expander("📌 7. Notes Techniques", expanded=False):
        st.markdown("""
        - **Calcul symbolique** : Si possible, l'app calcule la valeur exacte avec SymPy.
        - **Calcul numérique** : Si le calcul symbolique échoue, utilise des méthodes numériques.
        - **Performances** : Gauss-Legendre est rapide pour petit `n`, Monte Carlo lent.
        """)

    st.success("✅ **Prêt à utiliser l'application !** Entrez votre fonction, ajustez les paramètres et explorez les résultats.")

# Pied de page
st.sidebar.markdown("---")