"""
Testy dla algorytmów z solvers.py
Każdy test sprawdza poprawność implementacji algorytmu
"""

import sys
sys.path.append('.')

from solvers import (
    mcnaughton, knapsack_dp, mst_kruskal, huffman,
    weighted_late_jobs, moore_hodgson, list_scheduling,
    schrage, schrage_pmtn, carlier, brown_algorithm,
    Dinic, flow_with_bounds
)

def test_mcnaughton():
    """Test algorytmu McNaughtona (P|pmtn|Cmax)"""
    print("\n=== TEST: McNaughton Algorithm ===")
    
    # Test 1: Podstawowy przykład
    p_times = [2, 3, 4, 2]
    m = 3
    c_max, schedule = mcnaughton(p_times, m)
    
    print(f"Czasy zadań: {p_times}")
    print(f"Liczba maszyn: {m}")
    print(f"C_max = {c_max}")
    print(f"Oczekiwany C_max = max(4, 11/3) = 4")
    
    assert abs(c_max - 4.0) < 1e-6, "Błąd: C_max powinno być 4"
    
    # Sprawdź czy wszystkie zadania są zaszeregowane
    total_time = sum(p_times)
    scheduled_time = 0
    for machine in schedule:
        for task in machine:
            scheduled_time += (task['end'] - task['start'])
    
    assert abs(total_time - scheduled_time) < 1e-6, "Błąd: Nie wszystkie zadania zaszeregowane"
    
    # Test 2: Jedno długie zadanie
    p_times = [10, 1, 1, 1]
    m = 2
    c_max, _ = mcnaughton(p_times, m)
    assert abs(c_max - 10.0) < 1e-6, "C_max powinno być 10"
    
    print("✓ McNaughton: PASSED")


def test_knapsack_dp():
    """Test problemu plecakowego (DP po wartościach)"""
    print("\n=== TEST: Knapsack DP ===")
    
    # Test 1: Klasyczny przykład
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    
    best_val, items = knapsack_dp(weights, values, capacity)
    
    print(f"Wagi: {weights}")
    print(f"Wartości: {values}")
    print(f"Pojemność: {capacity}")
    print(f"Najlepsza wartość: {best_val}")
    print(f"Wybrane przedmioty (indeksy 1-based): {items}")
    
    # Optymalnie: przedmioty 2,4 (wagi 3+5=8, wartości 4+6=10)
    # To dokładnie wypełnia plecak!
    assert best_val == 10, f"Błąd: wartość powinna być 10, jest {best_val}"
    
    # Sprawdź czy wybrane przedmioty mieszczą się
    selected_weight = sum(weights[i-1] for i in items)
    selected_value = sum(values[i-1] for i in items)
    assert selected_weight <= capacity, "Przekroczona pojemność"
    assert selected_value == best_val, "Niezgodna wartość"
    
    # Test 2: Pusty plecak
    best_val, items = knapsack_dp([10], [5], 3)
    assert best_val == 0, "Pusty plecak powinien mieć wartość 0"
    
    print("✓ Knapsack DP: PASSED")


def test_mst_kruskal():
    """Test MST - Kruskal"""
    print("\n=== TEST: MST Kruskal ===")
    
    # Test: Graf z wykładu (6 wierzchołków)
    #   1---2
    #   |\ /|
    #   | X |
    #   |/ \|
    #   3---4
    
    n = 4
    edges = [
        (1, 2, 1),
        (1, 3, 3),
        (1, 4, 4),
        (2, 3, 5),
        (2, 4, 2),
        (3, 4, 6)
    ]
    
    weight, mst_edges = mst_kruskal(n, edges)
    
    print(f"Graf: {n} wierzchołków")
    print(f"Krawędzie wejściowe: {edges}")
    print(f"Waga MST: {weight}")
    print(f"Krawędzie MST: {mst_edges}")
    
    # MST powinno mieć n-1 krawędzi
    assert len(mst_edges) == n - 1, f"MST powinno mieć {n-1} krawędzi"
    
    # Sprawdź wagę (1+2+3 = 6)
    assert weight == 6, f"Waga MST powinna być 6, jest {weight}"
    
    print("✓ MST Kruskal: PASSED")


def test_huffman():
    """Test kodowania Huffmana"""
    print("\n=== TEST: Huffman Coding ===")
    
    # Test: Przykład z wykładu (str. 99)
    symbols = {'a': 0.1, 'b': 0.3, 'c': 0.4, 'd': 0.2}
    
    codes = huffman(symbols)
    
    print(f"Symbole i częstości: {symbols}")
    print(f"Kody Huffmana: {codes}")
    
    # Sprawdź czy wszystkie symbole mają kody
    assert set(codes.keys()) == set(symbols.keys()), "Nie wszystkie symbole zakodowane"
    
    # Sprawdź czy kody są binarne
    for symbol, code in codes.items():
        assert all(c in '01' for c in code), f"Kod {code} nie jest binarny"
    
    # Sprawdź czy kody są prefiksowe (żaden kod nie jest prefiksem innego)
    code_list = list(codes.values())
    for i, code1 in enumerate(code_list):
        for j, code2 in enumerate(code_list):
            if i != j:
                assert not code2.startswith(code1), f"Kod {code1} jest prefiksem {code2}"
    
    # Oblicz średnią długość kodu
    avg_length = sum(symbols[s] * len(codes[s]) for s in symbols)
    print(f"Średnia długość kodu: {avg_length:.2f}")
    
    # Z wykładu: dla a,b,c,d powinno wyjść ~1.9
    assert 1.8 <= avg_length <= 2.1, f"Średnia długość {avg_length} jest nieprawidłowa"
    
    print("✓ Huffman: PASSED")


def test_weighted_late_jobs():
    """Test ważonej liczby spóźnionych zadań"""
    print("\n=== TEST: Weighted Late Jobs (1||Σw_jU_j) ===")
    
    # Test: prosty przykład
    p_times = [2, 3, 4, 2]
    deadlines = [3, 5, 7, 4]
    weights = [5, 4, 3, 2]
    
    late_weight = weighted_late_jobs(p_times, deadlines, weights)
    
    print(f"Czasy zadań: {p_times}")
    print(f"Terminy: {deadlines}")
    print(f"Wagi: {weights}")
    print(f"Waga spóźnionych zadań: {late_weight}")
    
    # Sprawdź czy wynik jest sensowny
    total_weight = sum(weights)
    assert 0 <= late_weight <= total_weight, "Waga spóźnionych zadań poza zakresem"
    
    print("✓ Weighted Late Jobs: PASSED")


def test_moore_hodgson():
    """Test algorytmu Moore'a-Hodgsona"""
    print("\n=== TEST: Moore-Hodgson (1||ΣU_j) ===")
    
    # Test: Przykład klasyczny
    p_times = [4, 2, 6, 5, 3]
    deadlines = [8, 12, 16, 20, 22]
    
    late_count, late_tasks = moore_hodgson(p_times, deadlines)
    
    print(f"Czasy zadań: {p_times}")
    print(f"Terminy: {deadlines}")
    print(f"Liczba spóźnionych zadań: {late_count}")
    print(f"Spóźnione zadania (ID): {late_tasks}")
    
    # Sprawdź czy wynik jest sensowny
    assert 0 <= late_count <= len(p_times), "Liczba spóźnionych poza zakresem"
    assert len(late_tasks) == late_count, "Niezgodna liczba zadań na liście"
    
    # Test 2: Wszystkie zdążą
    p_times = [1, 1, 1]
    deadlines = [10, 10, 10]
    late_count, late_tasks = moore_hodgson(p_times, deadlines)
    assert late_count == 0, "Wszystkie zadania powinny zdążyć"
    assert len(late_tasks) == 0, "Lista spóźnionych powinna być pusta"
    
    print("✓ Moore-Hodgson: PASSED")


def test_list_scheduling():
    """Test List Scheduling"""
    print("\n=== TEST: List Scheduling (P||C_max) ===")
    
    # Test: Przykład z wykładu
    p_times = [7, 6, 5, 4, 3, 2, 1]
    m = 3
    
    c_max, assignments = list_scheduling(p_times, m)
    
    print(f"Czasy zadań: {p_times}")
    print(f"Liczba maszyn: {m}")
    print(f"C_max: {c_max}")
    print(f"Przypisania: {assignments}")
    
    # Dolne granica: max(max_p, sum/m) = max(7, 28/3) = 9.33...
    lower_bound = max(max(p_times), sum(p_times) / m)
    print(f"Dolne ograniczenie: {lower_bound:.2f}")
    
    assert c_max >= lower_bound - 1e-6, "C_max mniejsze niż dolne ograniczenie"
    
    # Sprawdź czy wszystkie zadania przypisane
    all_tasks = []
    for machine_tasks in assignments:
        all_tasks.extend(machine_tasks)
    assert len(all_tasks) == len(p_times), "Nie wszystkie zadania przypisane"
    
    print("✓ List Scheduling: PASSED")


def test_schrage():
    """Test algorytmu Schrage"""
    print("\n=== TEST: Schrage (1|r_j|L_max) ===")
    
    # Test: Przykład z zadaniami o różnych r_j
    tasks = [
        {'id': 1, 'r': 0, 'p': 4, 'q': 5},
        {'id': 2, 'r': 1, 'p': 3, 'q': 3},
        {'id': 3, 'r': 2, 'p': 2, 'q': 7},
        {'id': 4, 'r': 3, 'p': 1, 'q': 2}
    ]
    
    l_max, schedule = schrage(tasks)
    
    print(f"Zadania: {tasks}")
    print(f"L_max: {l_max}")
    print(f"Kolejność: {schedule}")
    
    # Sprawdź czy wszystkie zadania w harmonogramie
    assert len(schedule) == len(tasks), "Nie wszystkie zadania w harmonogramie"
    assert set(schedule) == {t['id'] for t in tasks}, "Błędne zadania w harmonogramie"
    
    print("✓ Schrage: PASSED")


def test_schrage_pmtn():
    """Test Schrage z preempcją"""
    print("\n=== TEST: Schrage PMTN (1|r_j,pmtn|L_max) ===")
    
    tasks = [
        {'id': 1, 'r': 0, 'p': 4, 'q': 5},
        {'id': 2, 'r': 1, 'p': 3, 'q': 8},
        {'id': 3, 'r': 2, 'p': 2, 'q': 3}
    ]
    
    l_max = schrage_pmtn(tasks)
    
    print(f"Zadania: {tasks}")
    print(f"L_max (preemptive): {l_max}")
    
    # L_max z preempcją powinno być <= L_max bez preempcji
    l_max_non_preemptive, _ = schrage(tasks)
    assert l_max <= l_max_non_preemptive, "Preemptive powinno być <= non-preemptive"
    
    print("✓ Schrage PMTN: PASSED")


def test_carlier():
    """Test algorytmu Carliera"""
    print("\n=== TEST: Carlier (1|r_j|L_max) ===")
    
    # Test: Mały przykład (Carlier jest kosztowny)
    tasks = [
        {'id': 1, 'r': 0, 'p': 3, 'q': 4},
        {'id': 2, 'r': 1, 'p': 2, 'q': 2},
        {'id': 3, 'r': 2, 'p': 1, 'q': 5}
    ]
    
    l_max, schedule = carlier(tasks)
    
    print(f"Zadania: {tasks}")
    print(f"Optymalny L_max: {l_max}")
    print(f"Kolejność: {schedule}")
    
    # Carlier powinien dać <= Schrage
    l_max_schrage, _ = schrage(tasks)
    assert l_max <= l_max_schrage, "Carlier powinien być <= Schrage"
    
    print("✓ Carlier: PASSED")


def test_brown_coloring():
    """Test algorytmu Browna dla kolorowania grafów"""
    print("\n=== TEST: Brown Coloring ===")
    
    # Test 1: Graf dwudzielny K_{2,2}
    n = 4
    edges = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    min_colors, coloring = brown_algorithm(n, edges)
    
    print(f"Graf dwudzielny K_{{2,2}}: {n} wierzchołków")
    print(f"Krawędzie: {edges}")
    print(f"Liczba kolorów: {min_colors}")
    print(f"Kolorowanie: {coloring}")
    
    assert min_colors == 2, "Graf dwudzielny powinien mieć χ=2"
    
    # Sprawdź poprawność kolorowania
    for u, v in edges:
        assert coloring[u] != coloring[v], f"Sąsiedzi {u},{v} mają ten sam kolor!"
    
    # Test 2: Klika K_5
    n = 5
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    min_colors, coloring = brown_algorithm(n, edges)
    
    print(f"\nKlika K_5: {n} wierzchołków")
    print(f"Liczba kolorów: {min_colors}")
    
    assert min_colors == 5, "Klika K_5 powinna mieć χ=5"
    
    # Test 3: Cykl C_5 (nieparzysty)
    n = 5
    edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    
    min_colors, coloring = brown_algorithm(n, edges)
    
    print(f"\nCykl C_5: {n} wierzchołków")
    print(f"Liczba kolorów: {min_colors}")
    
    assert min_colors == 3, "Cykl nieparzysty C_5 powinien mieć χ=3"
    
    print("✓ Brown Coloring: PASSED")


def test_dinic():
    """Test algorytmu Dinica dla maksymalnego przepływu"""
    print("\n=== TEST: Dinic Max Flow ===")
    
    # Test: Przykład z wykładu (str. 165)
    # Sieć z 6 wierzchołkami
    n = 6
    dinic = Dinic(n)
    
    # Dodaj krawędzie (u, v, capacity)
    dinic.add_edge(0, 1, 3)  # s->1
    dinic.add_edge(0, 2, 2)  # s->2
    dinic.add_edge(1, 3, 3)  # 1->3
    dinic.add_edge(2, 3, 1)  # 2->3
    dinic.add_edge(2, 4, 3)  # 2->4
    dinic.add_edge(3, 5, 2)  # 3->t
    dinic.add_edge(4, 5, 2)  # 4->t
    
    s, t = 0, 5
    max_flow = dinic.max_flow(s, t)
    
    print(f"Źródło: {s}, Ujście: {t}")
    print(f"Maksymalny przepływ: {max_flow}")
    
    # Sprawdź czy przepływ jest rozsądny
    assert max_flow == 4, f"Przepływ powinien być 4, jest {max_flow}"
    
    print("✓ Dinic: PASSED")


def test_circulation_lower_bounds():
    """Test cyrkulacji z dolnymi ograniczeniami"""
    print("\n=== TEST: Circulation with Lower Bounds ===")
    
    # Test: Prosty przykład
    n = 4
    s, t = 0, 3
    
    # edges: (u, v, lower_bound, capacity)
    edges = [
        (0, 1, 1, 3),  # krawędź z dolnym ograniczeniem 1
        (0, 2, 0, 2),
        (1, 3, 1, 2),
        (2, 3, 0, 3)
    ]
    
    feasible, max_flow = flow_with_bounds(n, s, t, edges)
    
    print(f"Źródło: {s}, Ujście: {t}")
    print(f"Krawędzie (u,v,low,cap): {edges}")
    print(f"Cyrkulacja możliwa: {feasible}")
    print(f"Maksymalny przepływ: {max_flow}")
    
    if feasible:
        assert max_flow > 0, "Jeśli cyrkulacja możliwa, przepływ > 0"
    
    print("✓ Circulation with Lower Bounds: PASSED")


def run_all_tests():
    """Uruchom wszystkie testy"""
    print("=" * 60)
    print("URUCHAMIANIE TESTÓW DLA SOLVERS.PY")
    print("=" * 60)
    
    tests = [
        test_mcnaughton,
        test_knapsack_dp,
        test_mst_kruskal,
        test_huffman,
        test_weighted_late_jobs,
        test_moore_hodgson,
        test_list_scheduling,
        test_schrage,
        test_schrage_pmtn,
        test_carlier,
        test_brown_coloring,
        test_dinic,
        test_circulation_lower_bounds
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__}: FAILED")
            print(f"  Błąd: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"PODSUMOWANIE: {passed} testów zaliczonych, {failed} niezaliczonych")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)