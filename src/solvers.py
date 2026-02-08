import heapq
import sys
from collections import deque, defaultdict
import math

# ==============================================================================
# 1. Algorytm McNaughtona (P|pmtn|Cmax)
# ==============================================================================
def mcnaughton(p_times, m):
    """
    Szeregowanie zadań podzielnych na m maszynach w celu minimalizacji Cmax.
    Złożoność: O(n)
    """
    n = len(p_times)
    total_time = sum(p_times)
    max_p = max(p_times) if p_times else 0
    
    # Cmax to maksimum z (średnie obciążenie, najdłuższe zadanie)
    c_max = max(max_p, total_time / m)
    
    schedule = [[] for _ in range(m)]
    current_machine = 0
    current_time = 0.0
    
    for i, p in enumerate(p_times):
        time_left = p
        task_id = i + 1
        
        while time_left > 1e-9:
            space_on_machine = c_max - current_time
            
            if time_left <= space_on_machine:
                # Zadanie mieści się w całości na obecnej maszynie
                schedule[current_machine].append({
                    'task': task_id, 
                    'start': current_time, 
                    'end': current_time + time_left
                })
                current_time += time_left
                time_left = 0
                
                # Jeśli maszyna pełna (z dokładnością float), idź do następnej
                if abs(current_time - c_max) < 1e-9:
                    current_machine += 1
                    current_time = 0.0
            else:
                # Zadanie trzeba podzielić
                schedule[current_machine].append({
                    'task': task_id, 
                    'start': current_time, 
                    'end': c_max
                })
                time_left -= space_on_machine
                current_machine += 1
                current_time = 0.0
                
    return c_max, schedule

# ==============================================================================
# 2. Problem Plecakowy (DP - wartości w kolumnach)
# ==============================================================================
def knapsack_dp(weights, values, capacity):
    """
    Problem plecakowy - metoda DP względem wartości (z wykładu).
    Użyteczna, gdy sumaryczna wartość jest niewielka.
    dp[v] = minimalna waga potrzebna do osiągnięcia wartości v.
    """
    n = len(weights)
    max_val = sum(values)
   
    # dp[i][v] = minimal weight to achieve value v using first i items
    dp = [[float('inf')] * (max_val + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    # choice[i][v] = True if item i was taken to achieve value v
    choice = [[False] * (max_val + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for val in range(max_val + 1):
            # do not take item i
            dp[i][val] = dp[i - 1][val]
            # take item i if possible
            if val >= v and dp[i - 1][val - v] + w < dp[i][val]:
                dp[i][val] = dp[i - 1][val - v] + w
                choice[i][val] = True

    print("\n[DEBUG] Tabela DP (Wiersze: przedmioty, Kolumny: łączna wartość, Komórki: minimalna waga)")
    # Nagłówek kolumn (wartości)
    print("      ", end="") 
    for v in range(max_val + 1):
        print(f"{v:3}", end=" ")
    print()
    print("      " + "----" * (max_val + 1))

    # Wiersze
    for i in range(n + 1):
        item_info = "START" if i == 0 else f"P{i}(w={weights[i-1]},v={values[i-1]})"
        print(f"{i:2} | ", end="") # Indeks przedmiotu
        for v in range(max_val + 1):
            val = dp[i][v]
            if val == float('inf'):
                print("inf", end=" ")
            else:
                # Jeśli waga > pojemność, oznaczamy kolorem lub gwiazdką (opcjonalnie)
                # Tutaj po prostu wypisujemy wagę
                print(f"{val:3}", end=" ")
        print(f"  <- {item_info}")
    # --------------------------------------------------------

    best_val = 0
    for v in range(max_val, -1, -1):
        if dp[n][v] <= capacity:
            best_val = v
            break
            
    # reconstruction          
    
    chosen_items = []

    v = best_val
    for i in range(n, 0, -1):
        if choice[i][v]:
            chosen_items.append(i)
            v -= values[i - 1]

    chosen_items.reverse()
    return best_val, chosen_items

# ==============================================================================
# 3. Minimalne Drzewo Rozpinające (Kruskal)
# ==============================================================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n + 1))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def mst_kruskal(n, edges):
    """
    edges: lista krotek (u, v, waga)
    """
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []
    
    for u, v, w in edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))
            
    return mst_weight, mst_edges

# ==============================================================================
# 4. Algorytm Huffmana
# ==============================================================================
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman(symbols):
    """
    symbols: lista krotek (znak, częstość) lub słownik
    """
    if isinstance(symbols, dict):
        symbols = list(symbols.items())
        
    heap = [HuffmanNode(char, freq) for char, freq in symbols]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(heap, merged)
        
    root = heap[0]
    codes = {}
    
    def generate_codes(node, current_code):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code
            return
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
        
    generate_codes(root, "")
    return codes

# ==============================================================================
# 5. Ważona liczba spóźnionych zadań (1 || Sum w_j U_j) - str. 106
# ==============================================================================
def weighted_late_jobs(p_times, deadlines, weights):
    """
    Implementacja za pomocą Programowania Dynamicznego.
    Problem jest NP-trudny (plecakowy).
    dp[t] = maksymalna waga zadań wykonanych o czasie, kończących się dokładnie w czasie t (lub <= t).
    """
    n = len(p_times)
    tasks = []
    for i in range(n):
        tasks.append({'p': p_times[i], 'd': deadlines[i], 'w': weights[i], 'id': i+1})
    
    # Sortujemy po terminach (EDD) - warunek konieczny dla DP w szeregowaniu
    tasks.sort(key=lambda x: x['d'])
    
    max_time = sum(p['p'] for p in tasks) # Górne ograniczenie czasu
    
    # dp[t] = maksymalna waga zadań, które kończą się w czasie <= t
    dp = [0] * (max_time + 1)
    
    for task in tasks:
        p = task['p']
        d = task['d']
        w = task['w']
        
        # Iterujemy od tyłu, jak w plecaku
        # Zadanie może się skończyć najpóźniej w d, ale też nie później niż max_time
        limit = min(max_time, d)
        
        for t in range(limit, p - 1, -1):
            if dp[t - p] + w > dp[t]:
                dp[t] = dp[t - p] + w
                
        # Propagacja maksimum w górę (opcjonalne w tym wariancie, 
        # ale przydatne, by dp[t] znaczyło "w czasie <= t")
        for t in range(1, max_time + 1):
             dp[t] = max(dp[t], dp[t-1])

    max_weight_on_time = dp[max_time]
    total_weight = sum(task['w'] for task in tasks)
    
    return total_weight - max_weight_on_time

# ==============================================================================
# 6. Liczba spóźnionych zadań (Moore-Hodgson) - str. 105
# ==============================================================================
def moore_hodgson(p_times, deadlines):
    """
    Minimalizacja liczby spóźnionych zadań (1 || U_j).
    Algorytm zachłanny O(n log n).
    """
    tasks = []
    for i in range(len(p_times)):
        tasks.append((p_times[i], deadlines[i], i + 1))
        
    # 1. Sortuj zadania wg terminów (EDD)
    tasks.sort(key=lambda x: x[1])
    
    current_time = 0
    scheduled_tasks = [] # Max-heap (przechowuje (-p, deadline, id))
    
    late_tasks = []
    
    for p, d, task_id in tasks:
        heapq.heappush(scheduled_tasks, (-p, d, task_id))
        current_time += p
        
        if current_time > d:
            # Jeśli przekroczyliśmy termin, wyrzucamy najdłuższe zadanie z dotychczasowych
            longest_p, longest_d, longest_id = heapq.heappop(scheduled_tasks)
            # longest_p jest ujemne, więc odejmując dodajemy
            current_time += longest_p 
            late_tasks.append(longest_id)
            
    num_late = len(late_tasks)
    return num_late, late_tasks

# ==============================================================================
# 7. Metoda Ścieżki Krytycznej (CPM)
# ==============================================================================
def cpm(n, durations, edges):
    """
    n: liczba wierzchołków
    durations: lista czasów trwania zadań (wierzchołków)
    edges: lista krotek (u, v) oznaczająca u -> v
    Zwraca: (ES, LS, Critical_Path_Length)
    """
    adj = defaultdict(list)
    in_degree = [0] * n
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
        
    # Sortowanie topologiczne
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        raise ValueError("Graf zawiera cykl!")
        
    # ES (Earliest Start)
    es = [0] * n
    # EF (Earliest Finish) = ES + duration
    ef = [0] * n
    
    for u in topo_order:
        ef[u] = es[u] + durations[u]
        for v in adj[u]:
            es[v] = max(es[v], ef[u])
            
    c_max = max(ef)
    
    # LS (Latest Start)
    ls = [c_max - durations[i] for i in range(n)] # Inicjalizacja na max possible
    lf = [c_max] * n
    
    for u in reversed(topo_order):
        # LF[u] = min(LS[v]) dla wszystkich v będących następcami u
        if adj[u]:
            lf[u] = min(ls[v] for v in adj[u])
        else:
            lf[u] = c_max
        ls[u] = lf[u] - durations[u]
        
    return es, ls, c_max

# ==============================================================================
# 8 & 9. Algorytm Dinica i Przepływy z ograniczeniami
# ==============================================================================
class Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.level = []

    def add_edge(self, u, v, capacity):
        # Forward edge: index w liście sąsiadów u to len(graph[u])
        # Backward edge: index w liście sąsiadów v to len(graph[v])
        self.graph[u].append([v, capacity, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def bfs(self, s, t):
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v, cap, rev_idx in self.graph[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)
        return self.level[t] >= 0

    def dfs(self, u, t, flow, ptr):
        if u == t or flow == 0:
            return flow
        for i in range(ptr[u], len(self.graph[u])):
            ptr[u] = i
            v, cap, rev_idx = self.graph[u][i]
            if self.level[v] == self.level[u] + 1 and cap > 0:
                pushed = self.dfs(v, t, min(flow, cap), ptr)
                if pushed > 0:
                    self.graph[u][i][1] -= pushed
                    self.graph[v][rev_idx][1] += pushed
                    return pushed
        return 0

    def max_flow(self, s, t):
        max_f = 0
        while self.bfs(s, t):
            ptr = [0] * self.n
            while True:
                pushed = self.dfs(s, t, float('inf'), ptr)
                if pushed == 0:
                    break
                max_f += pushed
        return max_f

def flow_with_bounds(n, s, t, edges):
    """
    edges: list of (u, v, lower_bound, capacity)
    Zwraca: (czy_możliwe, max_flow)
    """
    SS = n
    TT = n + 1
    dinic = Dinic(n + 2)

    balance = [0] * n

    # Budowa grafu z capacity = cap - low
    for u, v, low, cap in edges:
        dinic.add_edge(u, v, cap - low)
        balance[u] -= low
        balance[v] += low

    demand = 0
    for i in range(n):
        if balance[i] > 0:
            dinic.add_edge(SS, i, balance[i])
            demand += balance[i]
        elif balance[i] < 0:
            dinic.add_edge(i, TT, -balance[i])

    # krawędź domykająca obieg
    dinic.add_edge(t, s, float('inf'))

    # sprawdzamy istnienie cyrkulacji
    flow = dinic.max_flow(SS, TT)
    if flow != demand:
        return False, 0

    # usuwamy SS, TT (nie fizycznie, tylko logicznie)
    # szukamy max flow s -> t w grafie resztowym
    max_flow = dinic.max_flow(s, t)
    return True, max_flow

# ==============================================================================
# 10. Szeregowanie zadań na maszynach (List Scheduling)
# ==============================================================================
def list_scheduling(p_times, m):
    """
    Algorytm zachłanny LS dla P || Cmax.
    """
    # Kolejka priorytetowa maszyn: (moment_zakończenia, id_maszyny)
    machines = [(0, i) for i in range(m)]
    heapq.heapify(machines)
    
    task_assignments = [[] for _ in range(m)]
    
    for i, p in enumerate(p_times):
        finish_time, machine_id = heapq.heappop(machines)
        
        start_time = finish_time
        new_finish_time = start_time + p
        
        task_assignments[machine_id].append(i + 1)
        heapq.heappush(machines, (new_finish_time, machine_id))
        
    c_max = max(m[0] for m in machines) # Po pętli heap może nie być posortowany idealnie, ale max wyciągniemy
    # Lepiej:
    c_max = max(item[0] for item in machines)
    return c_max, task_assignments

# ==============================================================================
# 11. Algorytm Schrage / Carlier (Hipoteza dla stron 241-246)
# ==============================================================================
def schrage(tasks):
    """
    Algorytm Schrage dla 1 | r_j | L_max (lub pomocniczo w Carlier).
    Tasks: lista obiektów/słowników {r, p, q, id}
    q = czas stygnięcia (delivery time), L_max = max(C_j + q_j)
    Jeśli zadanie nie ma q, przyjmij q=0.
    """
    # Sortuj wg r_j
    N = sorted(tasks, key=lambda x: x['r'])
    G = [] # Gotowe zadania (kolejka priorytetowa wg q_j malejąco)
    
    t = 0
    k = 0
    u_max = -float('inf')
    schedule = []
    
    while G or k < len(N):
        while k < len(N) and N[k]['r'] <= t:
            # Dodaj do gotowych: w Pythonie heapq to min-heap, więc wrzucamy -q
            heapq.heappush(G, (-N[k]['q'], N[k]))
            k += 1
            
        if not G:
            t = N[k]['r']
            continue
            
        _, task = heapq.heappop(G)
        t += task['p']
        u_max = max(u_max, t + task['q'])
        schedule.append(task['id'])
        
    return u_max, schedule

def schrage_pmtn(tasks):
    """
    Preemptive Schrage
    1 | r_j, pmtn | L_max
    Zwraca: L_max (dolne ograniczenie)
    """
    tasks = [{'r': t['r'], 'p': t['p'], 'q': t['q']} for t in tasks]
    N = sorted(tasks, key=lambda x: x['r'])
    G = []

    t = 0
    L_max = 0
    current = None

    while G or N or current:
        while N and N[0]['r'] <= t:
            job = N.pop(0)
            heapq.heappush(G, (-job['q'], job))
            if current and job['q'] > current['q']:
                current['p'] -= (t - current['start'])
                t = job['r']
                if current['p'] > 0:
                    heapq.heappush(G, (-current['q'], current))
                current = None

        if not current:
            if not G:
                t = N[0]['r']
                continue
            _, current = heapq.heappop(G)
            current['start'] = t

        next_r = N[0]['r'] if N else float('inf')
        exec_time = min(current['p'], next_r - t)
        current['p'] -= exec_time
        t += exec_time

        if current['p'] == 0:
            L_max = max(L_max, t + current['q'])
            current = None

    return L_max

# ==============================================================================
# 12. Algorytm Browna (Kolorowanie Grafów)
# ==============================================================================
class BrownColoring:
    def __init__(self, n, edges):
        self.n = n
        self.adj = [set() for _ in range(n)]
        for u, v in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
            
        self.best_coloring = {}
        self.min_colors = n + 1
        self.colors = {} # current coloring

    def is_safe(self, node, color):
        for neighbor in self.adj[node]:
            if neighbor in self.colors and self.colors[neighbor] == color:
                return False
        return True

    def solve(self, node_idx=0, used_colors_count=0):
        # Branch & Bound
        if used_colors_count >= self.min_colors:
            return

        if node_idx == self.n:
            if used_colors_count < self.min_colors:
                self.min_colors = used_colors_count
                self.best_coloring = self.colors.copy()
            return

        # Próbuj pokolorować wierzchołek node_idx
        # Algorytm Browna: próbuj kolory 1..used_colors_count + 1
        for c in range(1, used_colors_count + 2):
            if self.is_safe(node_idx, c):
                self.colors[node_idx] = c
                new_count = max(used_colors_count, c)
                
                # Proste cięcie
                if new_count < self.min_colors:
                    self.solve(node_idx + 1, new_count)
                
                del self.colors[node_idx] # backtrack

def brown_algorithm(n, edges):
    solver = BrownColoring(n, edges)
    # Heurystyka na start: sortuj wierzchołki wg stopnia (opcjonalne, ale przyspiesza)
    # Tu wersja podstawowa (rekurencja po indeksach 0..n-1)
    solver.solve()
    return solver.min_colors, solver.best_coloring

# ==============================================================================
# 13. Algorytm Carliera (1 | r_j | L_max) – str. 241–246
# ==============================================================================
def carlier(tasks):
    """
    Algorytm Carliera dla problemu 1 | r_j | L_max.
    tasks: lista słowników {id, r, p, q}
    Zwraca: (L_max, kolejność zadań)
    """
    UB, pi = schrage(tasks)
    LB = schrage_pmtn(tasks)

    if LB >= UB:
        return UB, pi

    # wyznaczenie bloku krytycznego
    t = 0
    C = []
    for job_id in pi:
        job = next(j for j in tasks if j['id'] == job_id)
        t = max(t, job['r']) + job['p']
        C.append((job, t))

    b = max(range(len(C)), key=lambda i: C[i][1] + C[i][0]['q'])

    a = b
    while a > 0:
        if C[a - 1][1] == C[a][1] - C[a][0]['p']:
            a -= 1
        else:
            break

    c = None
    for i in range(a, b):
        if C[i][0]['q'] < C[b][0]['q']:
            c = i
    if c is None:
        return UB, pi

    block = C[c + 1 : b + 1]
    r_block = min(j['r'] for j, _ in block)
    p_block = sum(j['p'] for j, _ in block)
    q_block = min(j['q'] for j, _ in block)

    job_c = C[c][0]
    r_old, q_old = job_c['r'], job_c['q']

    # gałąź 1: modyfikacja r_c
    job_c['r'] = max(job_c['r'], r_block + p_block)
    UB1, _ = carlier(tasks)
    UB = min(UB, UB1)
    job_c['r'] = r_old

    # gałąź 2: modyfikacja q_c
    job_c['q'] = max(job_c['q'], q_block + p_block)
    UB2, _ = carlier(tasks)
    UB = min(UB, UB2)
    job_c['q'] = q_old

    return UB, pi

# ==============================================================================
# 14. Algorytm Floyda (Wszystkie pary najkrótszych ścieżek) - str. 150
# ==============================================================================
def floyd_warshall(n, adj_matrix):
    """
    Oblicza najkrótsze ścieżki pomiędzy wszystkimi parami wierzchołków.
    Wejście:
        n: liczba wierzchołków
        adj_matrix: macierz sąsiedztwa n x n, gdzie adj_matrix[i][j] to waga krawędzi.
                    Brak krawędzi powinien być oznaczony jako float('inf').
    Wyjście:
        D: macierz n x n z najkrótszymi odległościami.
    Złożoność: O(n^3)
    """
    # 1. Inicjalizacja macierzy D jako kopii macierzy wag (krok 2-3)
    # Tworzymy głęboką kopię, aby nie modyfikować oryginału
    D = [row[:] for row in adj_matrix]

    # 2. Ustawienie odległości wierzchołka do samego siebie na 0 (krok 4)
    for i in range(n):
        D[i][i] = 0

    # 3. Główna pętla algorytmu (kroki 5-8)
    # k - wierzchołek pośredni
    for k in range(n):
        # i - wierzchołek początkowy
        for i in range(n):
            # j - wierzchołek końcowy
            for j in range(n):
                # Relaksacja: czy przejście przez k jest krótsze?
                # Sprawdzamy, czy ścieżki istnieją (nie są nieskończonością), aby uniknąć błędów
                if D[i][k] != float('inf') and D[k][j] != float('inf'):
                    if D[i][j] > D[i][k] + D[k][j]:
                        D[i][j] = D[i][k] + D[k][j]

    return D
  
# ==============================================================================
# 15. Skojarzenia w grafach dwudzielnych (Algorytm Etykietowania) - str. 243-246
# ==============================================================================
def bipartite_matching_labeling(n_x, n_y, edges):
    """
    Znajduje maksymalne skojarzenie w grafie dwudzielnym metodą etykietowania.
    Implementacja oparta na procedurach scan_left / scan_right oraz stosie LIFO.
    
    Złożoność: O(nm)
    
    Wejście:
        n_x: liczba wierzchołków w zbiorze X (indeksy 0..n_x-1)
        n_y: liczba wierzchołków w zbiorze Y (indeksy 0..n_y-1)
        edges: lista krotek (u, v), gdzie u należy do X, v należy do Y
        
    Wyjście:
        size: liczność skojarzenia
        matching: lista krotek (u, v) należących do skojarzenia
    """
    # Budowa list sąsiedztwa dla zbioru X
    adj = [[] for _ in range(n_x)]
    for u, v in edges:
        if 0 <= u < n_x and 0 <= v < n_y:
            adj[u].append(v)

    # M: Aktualne skojarzenie
    # match_x[u] = v (u z X jest skojarzony z v z Y)
    # match_y[v] = u (v z Y jest skojarzony z u z X)
    match_x = [-1] * n_x
    match_y = [-1] * n_y

    def find_augmenting_path():
        # L_set: zbiór wolnych wierzchołków w X (zainicjowany na początku)
        # Stos (LIFO) do przechowywania wierzchołków do przetworzenia
        # Elementy na stosie to krotki: (wierzchołek, strona 'X' lub 'Y')
        stack = []
        
        # Słownik etykiet do odtwarzania ścieżki: labels[node] = parent_node
        # Klucze to krotki (id, 'X') lub (id, 'Y')
        labels = {}
        
        # Inicjalizacja stosu wolnymi wierzchołkami z X
        free_x = [u for u in range(n_x) if match_x[u] == -1]
        for u in free_x:
            stack.append((u, 'X'))
            labels[(u, 'X')] = None # Korzenie ścieżek
            
        visited = set()
        for u in free_x:
            visited.add((u, 'X'))

        while stack:
            curr_id, side = stack.pop() # LIFO behavior

            if side == 'X':
                # scan_left(u)
                # Przeglądamy sąsiadów u (wierzchołki v z Y)
                u = curr_id
                for v in adj[u]:
                    if (v, 'Y') not in visited:
                        visited.add((v, 'Y'))
                        labels[(v, 'Y')] = (u, 'X') # Etykietujemy v przez u
                        stack.append((v, 'Y'))      # Dodajemy do R (na stos)
            
            else: # side == 'Y'
                # scan_right(v)
                v = curr_id
                u_matched = match_y[v]
                
                if u_matched != -1:
                    # Jeśli v jest skojarzony z u_matched, idziemy po krawędzi skojarzenia
                    if (u_matched, 'X') not in visited:
                        visited.add((u_matched, 'X'))
                        labels[(u_matched, 'X')] = (v, 'Y') # Etykietujemy u przez v
                        stack.append((u_matched, 'X'))      # Dodajemy do L (na stos)
                else:
                    # Jeśli v jest wolny, znaleźliśmy ścieżkę powiększającą!
                    return v, labels
        
        return None, None

    # Główna pętla algorytmu
    while True:
        # 1. Szukaj ścieżki powiększającej
        end_v, labels = find_augmenting_path()
        
        # Jeśli nie znaleziono ścieżki, kończymy
        if end_v is None:
            break
            
        # 2. Powiększ skojarzenie (M