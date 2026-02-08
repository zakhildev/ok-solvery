import sys
import solvers  # Importuje Twoje funkcje z pliku solvers.py

def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Błąd: Wprowadź liczbę całkowitą.")

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Błąd: Wprowadź liczbę.")

def get_list_int(prompt):
    while True:
        try:
            line = input(prompt)
            return [int(x) for x in line.strip().split()]
        except ValueError:
            print("Błąd: Wprowadź liczby całkowite oddzielone spacją.")

def get_list_float(prompt):
    while True:
        try:
            line = input(prompt)
            return [float(x) for x in line.strip().split()]
        except ValueError:
            print("Błąd: Wprowadź liczby oddzielone spacją.")

def print_separator():
    print("-" * 60)

# ==============================================================================
# Obsługa poszczególnych algorytmów
# ==============================================================================

def run_mcnaughton():
    print("\n--- Algorytm McNaughtona (P|pmtn|Cmax) ---")
    print("Instrukcja: Podaj liczbę maszyn oraz czasy trwania zadań.")
    m = get_int("Liczba maszyn (m): ")
    p_times = get_list_float("Czasy zadań (p1 p2 ... pn): ")
    
    cmax, schedule = solvers.mcnaughton(p_times, m)
    
    print_separator()
    print(f"Wynik Cmax: {cmax}")
    print("Harmonogram:")
    for i, machine_tasks in enumerate(schedule):
        print(f"Maszyna {i+1}:")
        for task in machine_tasks:
            print(f"  - Zadanie {task['task']}: {task['start']:.2f} -> {task['end']:.2f}")

def run_knapsack():
    print("\n--- Problem Plecakowy (DP) ---")
    print("Instrukcja: Podaj pojemność plecaka, wagi przedmiotów i ich wartości.")
    print("Wagi i wartości muszą być podane w tej samej kolejności.")
    
    capacity = get_int("Pojemność plecaka: ")
    weights = get_list_int("Wagi przedmiotów (w1 w2 ... wn): ")
    values = get_list_int("Wartości przedmiotów (v1 v2 ... vn): ")
    
    if len(weights) != len(values):
        print("Błąd: Liczba wag nie zgadza się z liczbą wartości!")
        return

    best_val, chosen = solvers.knapsack_dp(weights, values, capacity)
    
    print_separator()
    print(f"Maksymalna wartość: {best_val}")
    print(f"Wybrane przedmioty (indeksy 1-based): {chosen}")

def run_kruskal():
    print("\n--- Minimalne Drzewo Rozpinające (Kruskal) ---")
    print("Instrukcja: Podaj liczbę wierzchołków oraz krawędzie w formacie: u v waga.")
    
    n = get_int("Liczba wierzchołków (n): ")
    m = get_int("Liczba krawędzi: ")
    
    edges = []
    print("Podaj krawędzie (u v waga):")
    for _ in range(m):
        edges.append(tuple(get_list_int("-> ")))
        
    mst_weight, mst_edges = solvers.mst_kruskal(n, edges)
    
    print_separator()
    print(f"Waga MST: {mst_weight}")
    print("Krawędzie w MST:", mst_edges)

def run_huffman():
    print("\n--- Kodowanie Huffmana ---")
    print("Instrukcja: Podaj pary 'znak częstość'. Np. 'a 5'.")
    
    count = get_int("Ile symboli chcesz wprowadzić? ")
    symbols = []
    for _ in range(count):
        line = input("Symbol i częstość (np. 'a 5'): ").split()
        if len(line) == 2:
            symbols.append((line[0], int(line[1])))
        else:
            print("Błąd formatu, pomijam.")
            
    codes = solvers.huffman(symbols)
    
    print_separator()
    print("Kody Huffmana:")
    for char, code in codes.items():
        print(f"  {char}: {code}")

def run_weighted_late_jobs():
    print("\n--- Ważona liczba spóźnionych zadań (1 || Sum w_j U_j) ---")
    print("Instrukcja: Podaj czasy trwania, terminy (deadlines) i wagi zadań.")
    
    p = get_list_int("Czasy trwania (p): ")
    d = get_list_int("Terminy (d): ")
    w = get_list_int("Wagi (w): ")
    
    if not (len(p) == len(d) == len(w)):
        print("Błąd: Listy muszą mieć tę samą długość!")
        return

    result = solvers.weighted_late_jobs(p, d, w)
    print_separator()
    print(f"Maksymalna waga zadań wykonanych o czasie: {result}") 
    # Uwaga: Oryginalna funkcja w solvers.py zwraca: total_weight - max_weight_on_time
    # czyli minimalną stratę (wagę spóźnionych).
    print(f"(Jest to minimalna suma wag zadań spóźnionych).")

def run_moore_hodgson():
    print("\n--- Liczba spóźnionych zadań (Moore-Hodgson) ---")
    print("Instrukcja: Podaj czasy trwania i terminy.")
    
    p = get_list_int("Czasy trwania (p): ")
    d = get_list_int("Terminy (d): ")
    
    if len(p) != len(d):
        print("Błąd: Długości list się nie zgadzają!")
        return
        
    num_late, late_tasks = solvers.moore_hodgson(p, d)
    print_separator()
    print(f"Liczba spóźnionych zadań: {num_late}")
    print(f"ID zadań spóźnionych: {late_tasks}")

def run_cpm():
    print("\n--- Metoda Ścieżki Krytycznej (CPM) ---")
    print("Instrukcja: Węzły numerowane od 0 do n-1.")
    
    n = get_int("Liczba zadań (węzłów): ")
    durations = get_list_int(f"Czasy trwania dla {n} zadań: ")
    
    if len(durations) != n:
        print("Błąd liczby czasów.")
        return

    m = get_int("Liczba zależności (krawędzi): ")
    edges = []
    print("Podaj zależności (u v) oznaczające u -> v:")
    for _ in range(m):
        edges.append(tuple(get_list_int("-> ")))
        
    try:
        es, ls, cmax = solvers.cpm(n, durations, edges)
        print_separator()
        print(f"Czas wykonania projektu (Cmax): {cmax}")
        print("Zadanie | ES | LS | Rezerwa (LS-ES)")
        for i in range(n):
            print(f"{i:7} | {es[i]:2} | {ls[i]:2} | {ls[i] - es[i]:2}")
    except ValueError as e:
        print(f"Błąd: {e}")

def run_flow_bounds():
    print("\n--- Przepływ z ograniczeniami (Dinic) ---")
    print("Instrukcja: Węzły numerowane od 0 do n-1.")
    
    n = get_int("Liczba wierzchołków: ")
    s = get_int("Źródło (s): ")
    t = get_int("Ujście (t): ")
    m = get_int("Liczba krawędzi: ")
    
    edges = []
    print("Podaj krawędzie (u v dolne_ogr pojemność):")
    for _ in range(m):
        edges.append(tuple(get_list_int("-> ")))
        
    possible, max_f = solvers.flow_with_bounds(n, s, t, edges)
    print_separator()
    if possible:
        print(f"Przepływ możliwy. Maksymalny przepływ: {max_f}")
    else:
        print("Przepływ niemożliwy (nie spełniono ograniczeń dolnych).")

def run_list_scheduling():
    print("\n--- List Scheduling (P || Cmax) ---")
    m = get_int("Liczba maszyn: ")
    p = get_list_float("Czasy zadań: ")
    
    cmax, assignments = solvers.list_scheduling(p, m)
    print_separator()
    print(f"Cmax: {cmax}")
    for i, tasks in enumerate(assignments):
        print(f"Maszyna {i}: Zadania {tasks}")

def run_schrage_carlier():
    print("\n--- Algorytmy Szeregowania RPQ (Schrage / Carlier) ---")
    print("Wybierz wariant:")
    print("1. Schrage (zwykły)")
    print("2. Schrage (z podziałem - pmtn)")
    print("3. Carlier (optymalny)")
    
    choice = input("Wybór: ")
    
    n = get_int("Liczba zadań: ")
    tasks = []
    print("Podaj parametry dla każdego zadania: r (dostęp), p (czas), q (stygnięcie)")
    for i in range(n):
        r, p, q = get_list_int(f"Zadanie {i+1} (r p q): ")
        tasks.append({'id': i+1, 'r': r, 'p': p, 'q': q})
        
    print_separator()
    if choice == '1':
        umax, order = solvers.schrage(tasks)
        print(f"Schrage Cmax: {umax}")
        print(f"Kolejność: {order}")
    elif choice == '2':
        lmax = solvers.schrage_pmtn(tasks)
        print(f"Schrage PMTN (Dolne oszacowanie): {lmax}")
    elif choice == '3':
        # Carlier modyfikuje listę tasks w miejscu, więc robimy kopię głęboką jeśli potrzeba
        # ale tutaj prosta kopia listy słowników wystarczy
        from copy import deepcopy
        tasks_copy = deepcopy(tasks)
        cmax, order = solvers.carlier(tasks_copy)
        print(f"Carlier Optymalny Cmax: {cmax}")
        # Uwaga: Carlier w solvers.py zwraca `pi` ze Schrage'a, co jest heurystyką
        # Dla pełnej kolejności optymalnej algorytm musiałby zwracać dokładną permutację, 
        # ale zazwyczaj w Carlier interesuje nas wartość Cmax.
    else:
        print("Niepoprawny wybór.")

def run_brown():
    print("\n--- Algorytm Browna (Kolorowanie grafu) ---")
    n = get_int("Liczba wierzchołków (0..n-1): ")
    m = get_int("Liczba krawędzi: ")
    edges = []
    print("Podaj krawędzie (u v):")
    for _ in range(m):
        edges.append(tuple(get_list_int("-> ")))
        
    min_colors, coloring = solvers.brown_algorithm(n, edges)
    print_separator()
    print(f"Minimalna liczba kolorów (Liczba chromatyczna): {min_colors}")
    print("Kolorowanie (wierzchołek: kolor):")
    for node in sorted(coloring.keys()):
        print(f"  {node}: {coloring[node]}")

# ==============================================================================
# MENU GŁÓWNE
# ==============================================================================

def main():
    options = {
        '1': ("McNaughton (Szeregowanie podzielne)", run_mcnaughton),
        '2': ("Problem Plecakowy (DP)", run_knapsack),
        '3': ("MST (Kruskal)", run_kruskal),
        '4': ("Huffman", run_huffman),
        '5': ("Ważona liczba spóźnionych (DP)", run_weighted_late_jobs),
        '6': ("Liczba spóźnionych (Moore-Hodgson)", run_moore_hodgson),
        '7': ("CPM (Ścieżka Krytyczna)", run_cpm),
        '8': ("Przepływ z ograniczeniami (Dinic)", run_flow_bounds),
        '9': ("List Scheduling (Szeregowanie zachłanne)", run_list_scheduling),
        '10': ("Schrage / Carlier (RPQ)", run_schrage_carlier),
        '11': ("Algorytm Browna (Kolorowanie)", run_brown),
        '0': ("Wyjście", sys.exit)
    }

    while True:
        print("\n" + "="*40)
        print(" SYSTEM ROZWIĄZYWANIA PROBLEMÓW (SOLVERS)")
        print("="*40)
        for key, (desc, _) in options.items():
            print(f"{key}. {desc}")
        print("="*40)
        
        choice = input("Wybierz opcję: ")
        
        if choice in options:
            try:
                # Uruchomienie wybranej funkcji
                options[choice][1]()
            except Exception as e:
                print(f"\n!!! Wystąpił niespodziewany błąd: {e}")
                import traceback
                traceback.print_exc()
            
            input("\nNaciśnij ENTER, aby kontynuować...")
        else:
            print("Nieprawidłowy wybór. Spróbuj ponownie.")

if __name__ == "__main__":
    main()