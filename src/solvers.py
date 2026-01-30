from collections import deque, defaultdict
import numpy as np
import heapq

def mcnaughton(processing_times, num_machines):
  """
  Rozwiązuje problem P|pmtn|Cmax algorytmem McNaughtona.
  
  Args:
    processing_times (list): Lista czasów wykonywania zadań (p_j)
    num_machines (int): Liczba dostępnych maszyn (m)
      
  Returns:
    float: Optymalny czas Cmax
    list: Harmonogram (lista list, gdzie każda podlista to zadania na danej maszynie)
      Format zadania: (id_zadania, start, koniec)
  """
  n = len(processing_times)
  total_work = sum(processing_times)
  max_job = max(processing_times)
  
  # Krok 1: Obliczenie optymalnego Cmax
  c_max = max(max_job, total_work / num_machines)
  
  schedule = [[] for _ in range(num_machines)]
  
  current_machine = 0
  current_time = 0.0
  
  # Krok 2-6: Układanie zadań
  for job_id, p in enumerate(processing_times):
    # Zadanie wchodzi w całości na obecną maszynę
    if current_time + p <= c_max + 1e-9: # 1e-9 dla błędu float
      schedule[current_machine].append((job_id, current_time, current_time + p))
      current_time += p
        
        # Jeśli idealnie wypełniliśmy maszynę, przeskocz do następnej
      if abs(current_time - c_max) < 1e-9:
        current_machine += 1
        current_time = 0.0
    else:
      # Zadanie trzeba podzielić (nie mieści się)
      part1 = c_max - current_time
      part2 = p - part1
      
      # Część 1: Dopełniamy obecną maszynę
      schedule[current_machine].append((job_id, current_time, c_max))
      
      # Przeskok na następną maszynę
      current_machine += 1
      if current_machine < num_machines:
        # Część 2: Reszta ląduje na początku nowej maszyny
        schedule[current_machine].append((job_id, 0.0, part2))
        current_time = part2
      else:
        # To teoretycznie nie powinno wystąpić przy poprawnym Cmax,
        # chyba że błędy zaokrągleń float są duże.
        pass
  return c_max, schedule

def knapsack(values, weights, capacity):
  """
  Alternatywny algorytm plecakowy (minimalizacja wagi dla danej wartości).
  Zgodny ze slajdami 67-68.
  
  Args:
    values (list): Lista wartości przedmiotów (w_i w notacji z wykładu)
    weights (list): Lista wag/rozmiarów przedmiotów (s_i w notacji z wykładu)
    capacity (int): Pojemność plecaka (b)
      
  Returns:
    int: Maksymalna wartość plecaka
    list: Indeksy wybranych przedmiotów
    np.array: Tabela DP (do wglądu)
  """
  n = len(values)
  max_possible_value = sum(values) # UB (Upper Bound)
  
  # Inicjalizacja tabeli nieskończonością (oznacza: nie da się uzyskać tej wartości)
  # Wiersze: 0..n (przedmioty), Kolumny: 0..UB (wartości)
  # g(i, j) w kodzie to dp[i][j]
  dp = np.full((n + 1, max_possible_value + 1), float('inf'))
  
  # Warunek brzegowy: Wartość 0 uzyskujemy wagą 0
  dp[0][0] = 0
  
  # Wypełnianie tabeli [cite: 724]
  for i in range(1, n + 1):
    v = values[i-1]   # wartość bieżącego przedmiotu
    w = weights[i-1]  # waga bieżącego przedmiotu
    
    for j in range(max_possible_value + 1):
      # Opcja 1: Nie bierzemy przedmiotu 'i' -> waga taka jak wcześniej
      not_taken = dp[i-1][j]
      
      # Opcja 2: Bierzemy przedmiot 'i' (jeśli celowana wartość j >= v)
      taken = float('inf')
      if j >= v:
        if dp[i-1][j-v] != float('inf'):
          taken = dp[i-1][j-v] + w
      
      # Wybieramy opcję dającą MNIEJSZĄ wagę
      dp[i][j] = min(not_taken, taken)

  # Szukanie wyniku: Największa wartość (j), dla której waga <= capacity
  best_value = 0
  for j in range(max_possible_value, -1, -1):
    if dp[n][j] <= capacity:
      best_value = j
      break
          
  # Odtwarzanie rozwiązania (backtracking)
  selected_items = []
  curr_val = best_value
  for i in range(n, 0, -1):
    # Jeśli waga jest inna niż wiersz wyżej przy tej samej wartości, 
    # to znaczy, że wzięliśmy przedmiot
    if dp[i][curr_val] != dp[i-1][curr_val]:
      selected_items.append(i-1) # Indeks 0-based
      curr_val -= values[i-1]

  return best_value, selected_items[::-1], dp

class KruskalMST:
  """
  Algorytm do drzewa rozpinającego
  """
  def __init__(self, adj_matrix):
    self.adj_matrix = np.array(adj_matrix)
    self.num_nodes = self.adj_matrix.shape[0]
    # Struktury dla Union-Find (zgodnie ze slajdami 105-106)
    self.parent = list(range(self.num_nodes))
    self.rank = [0] * self.num_nodes

  def find(self, i):
    """Znajduje reprezentanta zbioru (z kompresją ścieżki)."""
    if self.parent[i] != i:
      self.parent[i] = self.find(self.parent[i])
    return self.parent[i]

  def union(self, i, j):
    """Łączy dwa zbiory (union by rank)."""
    root_i = self.find(i)
    root_j = self.find(j)

    if root_i != root_j:
      # Dołączamy mniejsze drzewo do większego
      if self.rank[root_i] < self.rank[root_j]:
        self.parent[root_i] = root_j
      elif self.rank[root_i] > self.rank[root_j]:
        self.parent[root_j] = root_i
      else:
        self.parent[root_j] = root_i
        self.rank[root_i] += 1
      return True # Połączono
    return False # Już były w tym samym zbiorze (cykl)

  def solve(self):
    edges = []
    # 1. Konwersja macierzy na listę krawędzi: (waga, u, v)
    for r in range(self.num_nodes):
      for c in range(r + 1, self.num_nodes): # Tylko górny trójkąt macierzy
        weight = self.adj_matrix[r][c]
        if weight > 0: # Zakładamy, że 0 to brak krawędzi (lub waga > 0)
          edges.append((weight, r, c))
    
    # 2. Sortowanie krawędzi rosnąco wg wagi (kluczowe dla Kruskala)
    edges.sort(key=lambda x: x[0])
    
    mst_weight = 0
    mst_edges = []
    
    # 3. Główna pętla
    for weight, u, v in edges:
      # Jeśli wierzchołki są w różnych zbiorach, dodaj krawędź
      if self.union(u, v):
        mst_edges.append((u, v, weight))
        mst_weight += weight
            
    return mst_weight, mst_edges

def huffman(symbols_dict):
  """
  Buduje drzewo Huffmana i generuje kody.

  Args:
    symbols_dict (dict): Słownik {symbol: prawdopodobieństwo/częstość}
    
  Returns:
    dict: Słownik {symbol: kod_binarny}
    float: Średnia ważona długość kodu (L)
  """
  class _HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # Metoda potrzebna dla heapq do porównywania węzłów (sortowanie po freq)
    def __lt__(self, other):
        return self.freq < other.freq
  
  # 1. Tworzenie liści i wrzucanie na kopiec (kolejkę priorytetową)
  heap = [_HuffmanNode(char, freq) for char, freq in symbols_dict.items()]
  heapq.heapify(heap)

  # 2. Budowanie drzewa (łączenie dwóch najmniejszych)
  while len(heap) > 1:
    # Pobierz dwa węzły o najmniejszej częstości
    left_child = heapq.heappop(heap)
    right_child = heapq.heappop(heap)

    # Stwórz nowy węzeł wewnętrzny (suma wag)
    merged = _HuffmanNode(None, left_child.freq + right_child.freq)
    merged.left = left_child
    merged.right = right_child

    # Wrzuć z powrotem na kopiec
    heapq.heappush(heap, merged)

  # Ostatni element na kopcu to korzeń drzewa
  root = heap[0]

  # 3. Generowanie kodów (przejście DFS)
  codes = {}

  def generate_codes_recursive(node, current_code):
    if node is None:
        return

    # Jeśli to liść (ma symbol), zapisz kod
    if node.char is not None:
        codes[node.char] = current_code
        return

    # Rekurencja w lewo (dodaj '0') i w prawo (dodaj '1')
    generate_codes_recursive(node.left, current_code + "0")
    generate_codes_recursive(node.right, current_code + "1")

  generate_codes_recursive(root, "")

  # Obliczenie średniej długości (L) dla weryfikacji
  avg_length = sum(len(code) * symbols_dict[char] for char, code in codes.items())

  return codes, avg_length

def min_late_jobs_number(processing_times, due_dates):
  """
  Minimalizacja liczby spóźnionych zadań (Algorytm Moore'a-Hodgsona).
  Wersja poprawiona.
  """
  # Tworzymy listę zadań: (id, p, d)
  jobs = []
  for i in range(len(processing_times)):
    jobs.append({
      'id': i,
      'p': processing_times[i],
      'd': due_dates[i]
    })
  
  # 1. Sortowanie wg terminów (EDD) - rosnąco
  jobs.sort(key=lambda x: x['d'])
  
  current_time = 0
  # Kopiec przechowuje krotki: (-czas_trwania, id_zadania)
  # Przechowujemy tylko ID, żeby uniknąć błędu porównywania słowników
  scheduled_jobs_heap = [] 
  
  for job in jobs:
    # Dodajemy zadanie próbnie do harmonogramu
    # -job['p'] pozwala symulować max-heap (najdłuższe zadanie na wierzchu)
    heapq.heappush(scheduled_jobs_heap, (-job['p'], job['id'])) 
    current_time += job['p']
    
    # Jeśli zadanie się spóźniło (termin przekroczony)
    if current_time > job['d']:
      # Wyrzucamy najdłuższe zadanie z dotychczas zaplanowanych
      longest_p_neg, _ = heapq.heappop(scheduled_jobs_heap)
      longest_p = -longest_p_neg
      
      # Cofamy czas o długość wyrzuconego zadania
      current_time -= longest_p
          
  # Zbieramy wyniki z kopca (drugi element krotki to ID)
  on_time_indices = [jid for _, jid in scheduled_jobs_heap]
  on_time_indices.sort()
  
  num_late = len(jobs) - len(on_time_indices)
  
  return num_late, on_time_indices

def weighted_unit_tasks(weights, due_dates):
  """
  Minimalizacja ważonej liczby spóźnionych zadań dla pj=1.
  Problem: 1 | pj=1 | Sum wjUj

  Args:
    weights (list): Wagi zadań (w_j)
    due_dates (list): Terminy zakończenia (d_j)
    
  Returns:
    int: Suma wag spóźnionych zadań
    list: Harmonogram (tablica o długości n, gdzie schedule[t] to ID zadania w czasie t+1)
  """
  n = len(weights)
  jobs = []
  for i in range(n):
    jobs.append({
      'id': i,
      'w': weights[i],
      'd': due_dates[i]
    })
    
  # 1. Sortowanie wg wag malejąco (najważniejsze najpierw)
  jobs.sort(key=lambda x: x['w'], reverse=True)

  # Reprezentacja osi czasu (sloty). Czas biegnie 1..n
  # Używamy zbioru zajętych slotów dla szybkiego sprawdzania
  # (lub tablicy union-find dla dużych n, ale tu pętla wystarczy)
  schedule = [None] * n  # schedule[i] oznacza zadanie w slocie (czas i+1)
  late_jobs = []
  total_penalty = 0

  for job in jobs:
    assigned = False
    # 2. Szukamy wolnego slotu od min(d, n) w dół do 1
    # (indeksy w tablicy schedule: od min(d-1, n-1) do 0)
    start_search = min(job['d'], n) - 1
    
    for t in range(start_search, -1, -1):
      if schedule[t] is None:
        schedule[t] = job['id']
        assigned = True
        break
    
    if not assigned:
      late_jobs.append(job['id'])
      total_penalty += job['w']
        
  # Wypełniamy puste sloty zadaniami spóźnionymi (opcjonalne, dla pełnego harmonogramu)
  late_idx = 0
  for t in range(n):
    if schedule[t] is None and late_idx < len(late_jobs):
      schedule[t] = late_jobs[late_idx]
      late_idx += 1
        
  return total_penalty, schedule

def cpm(num_nodes, edges):
  """
  Metoda Ścieżki Krytycznej (CPM).
  Oblicza ES (alpha), LS (beta) i znajduje ścieżkę krytyczną.
  
  Args:
    num_nodes (int): Liczba węzłów (zdarzeń)
    edges (list): Lista krotek (u, v, waga), gdzie u->v to zadanie o czasie 'waga'
      
  Returns:
    tuple: (czas_projektu, lista_węzłów_krytycznych, tabela_ES, tabela_LS)
  """
  # Budowanie grafu i stopnie wejściowe (do sortowania topologicznego)
  adj = defaultdict(list)
  rev_adj = defaultdict(list) # Graf odwrócony do fazy "w tył"
  in_degree = [0] * num_nodes
  
  for u, v, w in edges:
    adj[u].append((v, w))
    rev_adj[v].append((u, w))
    in_degree[v] += 1
      
  # --- KROK 1: Sortowanie topologiczne (Algorytm Kahna) ---
  topo_order = []
  queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
  
  while queue:
    u = queue.popleft()
    topo_order.append(u)
    for v, w in adj[u]:
      in_degree[v] -= 1
      if in_degree[v] == 0:
        queue.append(v)
              
  if len(topo_order) != num_nodes:
    raise ValueError("Graf zawiera cykl! CPM działa tylko dla DAG.")

  # --- KROK 2: Faza w przód (Obliczanie ES / alpha) ---
  # ES[j] = max(ES[i] + waga(i, j)) dla wszystkich poprzedników i
  es = [-float('inf')] * num_nodes
  
  # Zakładamy, że węzły startowe (bez poprzedników) mają czas 0
  # (Chyba że wszystkie mają in_degree > 0, co w DAGu niemożliwe)
  for u in topo_order:
    if es[u] == -float('inf'): es[u] = 0 # Start
    
    for v, w in adj[u]:
      if es[u] + w > es[v]:
        es[v] = es[u] + w
              
  project_duration = max(es) # Czas całego projektu (CP)

  # --- KROK 3: Faza w tył (Obliczanie LS / beta) ---
  # LS[j] = min(LS[k] - waga(j, k)) dla wszystkich następników k
  ls = [float('inf')] * num_nodes

  # Węzły końcowe (bez następników) mają LS = czas_projektu
  for u in range(num_nodes):
    if not adj[u]:  # brak krawędzi wychodzących
      ls[u] = project_duration

  # Przechodzimy w odwrotnej kolejności topologicznej
  for u in reversed(topo_order):
    for v, w in adj[u]:
      if ls[v] - w < ls[u]:
        ls[u] = ls[v] - w

  # --- KROK 4: Identyfikacja ścieżki krytycznej ---
  # Węzeł jest krytyczny, jeśli ES == LS (luz == 0)
  critical_nodes = []
  for i in range(num_nodes):
    if abs(es[i] - ls[i]) < 1e-9: # Porównanie float
      critical_nodes.append(i)

  # Sortujemy, żeby ładnie wyglądało
  critical_nodes.sort()
          
  return project_duration, critical_nodes, es, ls

class Dinic:
  class Edge:
    def __init__(self, v, capacity, flow, rev_index):
      self.v = v              # Dokąd prowadzi krawędź
      self.capacity = capacity # Przepustowość (Upper Bound)
      self.flow = flow        # Aktualny przepływ
      self.rev_index = rev_index # Indeks krawędzi powrotnej w liście sąsiada

  def __init__(self, num_nodes):
    self.n = num_nodes
    self.graph = [[] for _ in range(num_nodes)]
    self.level = []

  def add_edge(self, u, v, capacity):
    """Dodaje krawędź skierowaną u -> v o danej przepustowości."""
    # Krawędź "w przód": capacity=C, flow=0
    forward = self.Edge(v, capacity, 0, len(self.graph[v]))
    # Krawędź "w tył" (rezydualna): capacity=0, flow=0
    backward = self.Edge(u, 0, 0, len(self.graph[u]))
    
    self.graph[u].append(forward)
    self.graph[v].append(backward)

  def _bfs(self, s, t):
    """Buduje sieć warstwową (Level Graph). Zwraca True jeśli t jest osiągalne."""
    self.level = [-1] * self.n
    self.level[s] = 0
    queue = [s]
    
    while queue:
      u = queue.pop(0)
      for edge in self.graph[u]:
        # Jeśli krawędź ma wolną przepustowość i wierzchołek nieodwiedzony
        if edge.capacity - edge.flow > 0 and self.level[edge.v] < 0:
          self.level[edge.v] = self.level[u] + 1
          queue.append(edge.v)
    
    return self.level[t] >= 0

  def _dfs(self, u, t, pushed_flow, ptr):
    """Szuka potoku blokującego w sieci warstwowej."""
    if pushed_flow == 0 or u == t:
      return pushed_flow
    
    # ptr[u] to optymalizacja: pamiętamy, które krawędzie już nasyciliśmy,
    # żeby ich nie sprawdzać ponownie w tej samej fazie.
    for i in range(ptr[u], len(self.graph[u])):
      ptr[u] = i # Aktualizujemy wskaźnik
      edge = self.graph[u][i]
      
      # Warunek Dinica: idziemy tylko do następnej warstwy (level + 1)
      # i tylko jeśli jest miejsce w krawędzi
      if self.level[u] + 1 != self.level[edge.v] or edge.capacity - edge.flow == 0:
        continue
      
      # Rekurencyjnie pchamy tyle, ile się da (min z obecnego push i wolnego miejsca)
      tr = self._dfs(edge.v, t, min(pushed_flow, edge.capacity - edge.flow), ptr)
      
      if tr == 0:
        continue
      
      # Aktualizacja przepływów (forward i backward)
      edge.flow += tr
      self.graph[edge.v][edge.rev_index].flow -= tr
      
      return tr
        
    return 0

  def solve_max_flow(self, s, t):
    max_flow = 0
    # Dopóki da się zbudować sieć warstwową (dojść do ujścia)
    while self._bfs(s, t):
      ptr = [0] * self.n # Reset wskaźników dla każdej fazy
      while True:
        pushed = self._dfs(s, t, float('inf'), ptr)
        if pushed == 0:
          break
        max_flow += pushed
            
    return max_flow

def scheduling_lpt(processing_times, num_machines):
  """
  Algorytm LPT (Longest Processing Time) dla problemu P||Cmax.
  
  Args:
    processing_times (list): Czasy wykonywania zadań
    num_machines (int): Liczba maszyn
      
  Returns:
    float: Maksymalny czas zakończenia (Cmax)
    list: Harmonogram (lista list, assignments[i] to zadania na maszynie i)
  """
  # 1. Zachowujemy oryginalne indeksy zadań: (czas, oryginalny_id)
  jobs = []
  for i, p in enumerate(processing_times):
    jobs.append((p, i))
      
  # 2. Sortujemy zadania malejąco wg czasu (Longest Processing Time)
  jobs.sort(key=lambda x: x[0], reverse=True)
  
  # 3. Inicjalizacja maszyn (kopiec min-heap do szybkiego szukania najmniej obciążonej)
  # W kopcu trzymamy krotki: (aktualne_obciążenie, numer_maszyny)
  machines_heap = [(0, i) for i in range(num_machines)]
  heapq.heapify(machines_heap)
  
  # Struktura do zapisu wyniku: assignments[nr_maszyny] = [(id_zadania, start, koniec), ...]
  assignments = [[] for _ in range(num_machines)]
  
  for p, job_id in jobs:
    # Pobierz najmniej obciążoną maszynę
    current_load, machine_id = heapq.heappop(machines_heap)
    
    start_time = current_load
    end_time = current_load + p
    
    # Przypisz zadanie
    assignments[machine_id].append({
      'id': job_id,
      'start': start_time,
      'end': end_time,
      'duration': p
    })
    
    # Zaktualizuj obciążenie maszyny i wrzuć z powrotem na kopiec
    new_load = end_time
    heapq.heappush(machines_heap, (new_load, machine_id))
      
  # Oblicz Cmax (największe obciążenie spośród maszyn)
  c_max = max(load for load, _ in machines_heap)
  
  return c_max, assignments

class BipartiteMatching:
  def __init__(self, bi_adj_matrix):
    """
    Args:
        bi_adj_matrix: Macierz (N x M), gdzie wiersze to zbiór U, kolumny to zbiór V.
                        1 oznacza krawędź, 0 brak.
    """
    self.graph = np.array(bi_adj_matrix)
    self.num_left = self.graph.shape[0]  # Liczba wierzchołków w lewej grupie
    self.num_right = self.graph.shape[1] # Liczba wierzchołków w prawej grupie
    
    # Tablica przechowująca, z kim skojarzony jest wierzchołek z prawej grupy
    # match_right[v] = u (oznacza, że v z prawej jest połączony z u z lewej)
    # -1 oznacza, że jest wolny
    self.match_right = [-1] * self.num_right

  def _dfs(self, u, visited):
    """
    Przeszukiwanie w głąb szukające ścieżki powiększającej dla wierzchołka u.
    """
    for v in range(self.num_right):
      # Jeśli istnieje krawędź u-v I wierzchołek v nie był jeszcze odwiedzony w tej iteracji
      if self.graph[u][v] > 0 and not visited[v]:
        visited[v] = True
        
        # Jeśli v jest wolny LUB osoba zajmująca v (match_right[v]) może znaleźć innego partnera
        if self.match_right[v] < 0 or self._dfs(self.match_right[v], visited):
          self.match_right[v] = u
          return True
    return False

  def solve(self):
    match_count = 0
    self.match_right = [-1] * self.num_right # Reset
    
    for u in range(self.num_left):
      # Dla każdego wierzchołka z lewej szukamy możliwości skojarzenia
      visited = [False] * self.num_right
      if self._dfs(u, visited):
        match_count += 1
            
    # Przygotowanie wyników (lista par)
    pairs = []
    for v in range(self.num_right):
      if self.match_right[v] != -1:
        pairs.append((self.match_right[v], v))
    
    # Sortowanie par dla czytelności (wg lewej strony)
    pairs.sort(key=lambda x: x[0])
    
    return match_count, pairs
  
class BrownColoring:
  def __init__(self, adj_matrix):
    """
    Inicjalizacja solvera Algorytmu Browna.
    Args:
        adj_matrix: Macierz sąsiedztwa (NxN), gdzie 1 oznacza krawędź.
    """
    self.adj = np.array(adj_matrix)
    self.n = len(adj_matrix)
    
    # Upper Bound (UB) - najlepsza znana liczba kolorów.
    # Na początku pesymistycznie zakładamy n+1.
    self.ub = self.n + 1
    
    # Przechowuje najlepsze znalezione przypisanie kolorów
    self.best_coloring = np.zeros(self.n, dtype=int)
    
    # Tablica bieżących kolorów (indeksowana wg posortowanych wierzchołków)
    self.current_coloring = np.zeros(self.n, dtype=int)
    
    # Sortowanie wierzchołków malejąco wg stopnia (heurystyka LF)
    # To kluczowe ulepszenie dla Algorytmu Browna.
    degrees = np.sum(self.adj, axis=1)
    self.order = np.argsort(degrees)[::-1]
    
    # Mapowanie odwrotne do sprawdzania sąsiedztwa w oryginalnej macierzy
    self.orig_indices = {sorted_idx: orig_idx for sorted_idx, orig_idx in enumerate(self.order)}

  def is_safe(self, node_idx, color):
    """
    Sprawdza, czy można nadać dany kolor wierzchołkowi 'node_idx'
    (względem już pokolorowanych wcześniejszych sąsiadów).
    """
    original_u = self.order[node_idx]
    
    # Sprawdzamy tylko wierzchołki już odwiedzone (indeksy < node_idx)
    for i in range(node_idx):
      if self.current_coloring[i] == color:
        original_v = self.order[i]
        # Sprawdzenie w oryginalnej macierzy sąsiedztwa
        if self.adj[original_u][original_v] == 1:
          return False
    return True

  def _brown_recursive(self, k, q):
    """
    Rekurencyjna implementacja logiki Forward/Backtrack.
    k: indeks aktualnie kolorowanego wierzchołka (w posortowanej tablicy)
    q: liczba użytych dotąd kolorów (max color index użyty w 0..k-1)
    """
    # Jeśli bieżące rozwiązanie już wymaga tyle samo lub więcej kolorów co najlepsze znane,
    # to odcinamy tę gałąź (PRUNING) - istota algorytmu Browna.
    if q >= self.ub:
      return

    # Jeśli dotarliśmy do końca (wszystkie wierzchołki pokolorowane)
    if k == self.n:
      if q < self.ub:
        self.ub = q
        self.best_coloring = self.current_coloring.copy()
      return

    # Próbujemy kolorów od 1 do q+1.
    # Algorytm Browna wymusza sprawdzanie kolorów w kolejności rosnącej.
    # Możemy wprowadzić nowy kolor (q+1) tylko jeśli nie da się użyć 1..q.
    for c in range(1, q + 2):
      # Kolejne cięcie: jeśli lokalny kolor c osiągnie UB, to nie ma sensu go używać,
      # bo i tak nie poprawimy wyniku.
      if c >= self.ub:
        break

      if self.is_safe(k, c):
        self.current_coloring[k] = c
        
        # Obliczamy nowe q (czy wprowadziliśmy nowy, wyższy kolor?)
        new_q = max(q, c)
        
        # Krok w przód (Forward)
        self._brown_recursive(k + 1, new_q)
        
        # Backtracking (cofnięcie przypisania przy powrocie z rekurencji)
        self.current_coloring[k] = 0

  def solve(self):
    """Uruchamia algorytm i zwraca (liczba_chromatyczna, mapowanie_kolorów)."""
    # Startujemy od wierzchołka 0, liczba użytych kolorów = 0
    self._brown_recursive(0, 0)
    
    # Odtworzenie wyniku w oryginalnej kolejności wierzchołków
    final_result = {}
    for sorted_idx, color in enumerate(self.best_coloring):
      original_idx = self.order[sorted_idx]
      final_result[original_idx] = color
        
    # Zwracamy posortowaną listę kolorów (dla wierzchołków 0, 1, 2...)
    sorted_colors = [final_result[i] for i in range(self.n)]
    return self.ub, sorted_colors

def flow_with_lower_bounds(num_nodes, edges, source, sink):
  """
  Przepływ w sieci z dolnymi i górnymi ograniczeniami.

  Args:
    num_nodes (int): liczba węzłów
    edges (list): lista krotek (u, v, lower, upper)
    source (int): źródło
    sink (int): ujście

  Returns:
    bool: czy istnieje dopuszczalny przepływ
    int: wartość przepływu (jeśli istnieje)
  """

  # Graf pomocniczy z super-źródłem i super-ujściem
  super_source = num_nodes
  super_sink = num_nodes + 1
  dinic = Dinic(num_nodes + 2)

  balance = [0] * (num_nodes + 2)

  # 1. Redukcja: usuwamy dolne ograniczenia
  for u, v, lower, upper in edges:
    # Nowa przepustowość = upper - lower
    dinic.add_edge(u, v, upper - lower)
    balance[u] -= lower
    balance[v] += lower

  # 2. Dodanie krawędzi z super-źródła / do super-ujścia
  for i in range(num_nodes):
    if balance[i] > 0:
      dinic.add_edge(super_source, i, balance[i])
    elif balance[i] < 0:
      dinic.add_edge(i, super_sink, -balance[i])

  # 3. Krawędź sink -> source o nieskończonej przepustowości
  INF = 10**18
  dinic.add_edge(sink, source, INF)

  # 4. Sprawdzenie istnienia przepływu
  total_demand = sum(b for b in balance if b > 0)
  flow = dinic.solve_max_flow(super_source, super_sink)

  if flow != total_demand:
    return False, 0

  # 5. Obliczenie rzeczywistego przepływu source -> sink
  # (usuwamy sztuczne węzły, liczymy przepływ przez krawędź sink->source)
  result_flow = 0
  for edge in dinic.graph[sink]:
    if edge.v == source:
      result_flow = edge.flow
      break

  return True, result_flow
