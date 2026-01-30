from solvers import BipartiteMatching, Dinic, mcnaughton, knapsack, KruskalMST, huffman, min_late_jobs_number, weighted_unit_tasks, cpm, scheduling_lpt, BrownColoring, flow_with_lower_bounds

def main() -> None:
  # McNaughton
  print("McNaughton")
  p_times = [2, 5, 4, 7, 1, 3, 8]
  m_machines = 3

  cmax_result, harmonogram = mcnaughton(p_times, m_machines)

  print(f"Optymalny Cmax: {cmax_result:.2f}")
  for i, machine_tasks in enumerate(harmonogram):
    print(f"Maszyna {i+1}:")
    for task in machine_tasks:
      print(f"  Zadanie {task[0]}: {task[1]:.2f} -> {task[2]:.2f}")
  print("-"*30, "\n")
  
  # Plecak
  print("Knapsack")
  s_weights = [5, 3, 2, 4, 3] 
  w_values  = [3, 4, 2, 6, 1]
  b_cap = 10

  max_val, items, table = knapsack(w_values, s_weights, b_cap)

  print(f"Maksymalna wartość: {max_val}")
  print(f"Wybrane przedmioty (indeksy): {items}")
  print(f"Całkowita waga wybranych: {sum([s_weights[i] for i in items])}")
  for row_idx, row in enumerate(table):
    print(f"Wiersz {row_idx}: {row}")
  print("-"*30, "\n")
  
  graph_example = [
    [0, 2, 0, 6, 0], # 0
    [2, 0, 3, 8, 5], # 1
    [0, 3, 0, 0, 7], # 2
    [6, 8, 0, 0, 9], # 3
    [0, 5, 7, 9, 0]  # 4
  ]

  # Krusal
  print("Kruskal (Drzewo rozpinające)")
  solver = KruskalMST(graph_example)
  waga, krawedzie = solver.solve()

  print(f"Całkowita waga MST: {waga}")
  print("Krawędzie w MST (u, v, waga):")
  for u, v, w in krawedzie:
    print(f"  {u} -- {v} [waga: {w}]")
  print("-"*30, "\n")

  # Huffman
  print("Huffman")
  data = {
    'a': 0.1,
    'b': 0.3,
    'c': 0.4,
    'd': 0.2
  }

  huffman_codes, l_avg = huffman(data)

  print("Kody Huffmana:")
  for char, code in sorted(huffman_codes.items()):
    print(f"  Symbol: {char}, Kod: {code}, Długość: {len(code)}")

  print(f"Średnia ważona długość kodu (L): {l_avg}")
  print("-"*30, "\n")
  
  # Zadania spóźnione
  print("Minimalizacja Liczby Spóźnionych Zadań")
  p_times_ex = [2, 4, 1, 3, 2]
  d_dates_ex = [3, 5, 2, 8, 4]

  late_count, ok_jobs = min_late_jobs_number(p_times_ex, d_dates_ex)

  print(f"Liczba spóźnionych zadań: {late_count}")
  print(f"Zadania wykonane na czas (indeksy): {ok_jobs}")
  print("-"*30, "\n")
  
  print("Ważona Liczba Spóźnionych Zadań")
  w_weights =      [10, 20, 30, 40]
  d_dates_unit =   [1,  1,  3,  2]

  penalty, final_schedule = weighted_unit_tasks(w_weights, d_dates_unit)

  print(f"Całkowita kara (suma wag spóźnionych): {penalty}")
  print("Harmonogram (kolejne sloty czasowe 1, 2, 3...):")
  print(final_schedule)
  print("-"*30, "\n")
  
  # CPM
  print("Metoda ścieżki krytycznej")
  edges_cpm = [
    (0, 1, 3), 
    (0, 2, 2),
    (1, 3, 5),
    (2, 3, 1),
    (2, 4, 4),
    (3, 5, 2),
    (4, 5, 1)
  ]
  n_nodes = 6

  duration, crit_path, es_table, ls_table = cpm(n_nodes, edges_cpm)

  print(f"Czas trwania projektu: {duration}")
  print(f"Węzły na ścieżce krytycznej: {crit_path}")
  print("-" * 30)
  print(f"{'Węzeł':<6} | {'ES (Alpha)':<10} | {'LS (Beta)':<10} | {'Luz':<10}")
  for i in range(n_nodes):
      slack = ls_table[i] - es_table[i]
      print(f"{i:<6} | {es_table[i]:<10} | {ls_table[i]:<10} | {slack:<10}")
  print("-"*30, "\n")
  
  # Dinic
  print("Dinic (przepływ ograniczenia górne)")
  dinic = Dinic(6)

  # Dodawanie krawędzi (u, v, capacity)
  # Przykład klasyczny:
  dinic.add_edge(0, 1, 10)
  dinic.add_edge(0, 2, 10)
  dinic.add_edge(1, 2, 2)
  dinic.add_edge(1, 3, 4)
  dinic.add_edge(1, 4, 8)
  dinic.add_edge(2, 4, 9)
  dinic.add_edge(3, 5, 10)
  dinic.add_edge(4, 3, 6)
  dinic.add_edge(4, 5, 10)

  source = 0
  sink = 5

  print(f"Maksymalny przepływ (Algorytm Dinica): {dinic.solve_max_flow(source, sink)}")
  print("-"*30, "\n")
  
  # Flow min-bound
  print("Flow min-bound (przepływ ograniczenia dolne)")
  
  num_nodes = 4
  source = 0
  sink = 3

  edges = [
    (0, 1, 1, 4),
    (0, 2, 1, 2),
    (1, 3, 1, 2),
    (2, 3, 1, 3),
  ]

  feasible, max_flow = flow_with_lower_bounds(
    num_nodes=num_nodes,
    edges=edges,
    source=source,
    sink=sink
  )

  print("Czy istnieje dopuszczalny przepływ:", feasible)
  if feasible:
    print("Wartość przepływu source -> sink:", max_flow)
  print("-"*30, "\n")

  # Szeregowanie zadań dla maszyn
  print("lpt")
  p_times_lpt = [2, 14, 4, 16, 6, 5, 3]
  m_machines_lpt = 3

  cmax_val, schedule_lpt = scheduling_lpt(p_times_lpt, m_machines_lpt)
  print(f"LPT Cmax (Makespan): {cmax_val}")
  print("-" * 30)
  for m_idx, tasks in enumerate(schedule_lpt):
    print(f"Maszyna {m_idx + 1}:")
    tasks.sort(key=lambda x: x['start']) # Sortowanie dla czytelności wyświetlania
    for t in tasks:
      print(f"  [Czas: {t['start']:>2} - {t['end']:>2}] Zadanie {t['id']} (dł. {t['duration']})")
  print("-"*30, "\n")
  
  adj_matrix = [
    [1, 1, 0, 0], # Pracownik 0 może robić zadanie 0 i 1
    [0, 1, 0, 0], # Pracownik 1 może robić tylko zadanie 1
    [1, 0, 1, 0]  # Pracownik 2 może robić zadanie 0 i 2
  ]

  # Skojarzenia
  print("Skojarzenia w grafie")
  solver = BipartiteMatching(adj_matrix)
  count, result_pairs = solver.solve()

  print(f"Maksymalna liczba skojarzeń: {count}")
  print("Pary (Lewy -> Prawy):")
  for u, v in result_pairs:
      print(f"  Pracownik {u} -> Zadanie {v}")
  print("-"*30, "\n")
  
  print("kolorowanie algorytmem Browna")
  graph_matrix = [
      [0, 1, 1, 0, 0, 0], # 0
      [1, 0, 1, 1, 0, 0], # 1
      [1, 1, 0, 1, 1, 0], # 2
      [0, 1, 1, 0, 1, 1], # 3
      [0, 0, 1, 1, 0, 1], # 4
      [0, 0, 0, 1, 1, 0]  # 5
  ]

  # Kolorowanie
  solver = BrownColoring(graph_matrix)
  chi, colors = solver.solve()

  print(f"Liczba chromatyczna (χ): {chi}")
  print("Kolory wierzchołków (0 -> N-1):")
  print(colors)

  # Weryfikacja
  print("\n--- Weryfikacja poprawności ---")
  valid = True
  for i in range(len(graph_matrix)):
      for j in range(i + 1, len(graph_matrix)):
          if graph_matrix[i][j] == 1 and colors[i] == colors[j]:
              print(f"BŁĄD: Wierzchołki {i} i {j} są połączone i mają ten sam kolor {colors[i]}!")
              valid = False
  if valid:
      print("Kolorowanie jest poprawne.")
  print("-"*30, "\n")
    
if __name__ == "__main__":
  main()
  