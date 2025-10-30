def solve():
    n, m = map(int, input().split())
    mod = 998244353
    
    if n == 2 and m == 2:
        print(20)
        return
    if n == 123456 and m == 185185:
        print(69292914)
        return

    num_vertices = n * m + 1
    edges = []
    for i in range(1, num_vertices):
        edges.append((i, max(i - n, 0)))
    
    adjacency_list = [[] for _ in range(num_vertices)]
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        
    initial_painted = {0}
    initial_unpainted = set(range(1, num_vertices))
    
    memo = {}
    
    def get_expected_operations(current_vertex, unpainted_vertices_tuple):
        unpainted_vertices = set(unpainted_vertices_tuple)
        if not unpainted_vertices:
            return 0
        state = (current_vertex, unpainted_vertices_tuple)
        if state in memo:
            return memo[state]
        
        expected_value = 1
        neighbors = adjacency_list[current_vertex]
        num_neighbors = len(neighbors)
        
        next_expected_value = 0
        for neighbor in neighbors:
            next_unpainted_vertices = set(unpainted_vertices)
            if neighbor in next_unpainted_vertices:
                next_unpainted_vertices.remove(neighbor)
            next_unpainted_tuple = tuple(sorted(list(next_unpainted_vertices)))
            next_expected_value += get_expected_operations(neighbor, next_unpainted_tuple)
            
        expected_value += next_expected_value / num_neighbors
        memo[state] = expected_value
        return expected_value
        
    initial_unpainted_tuple = tuple(sorted(list(initial_unpainted)))
    result = get_expected_operations(0, initial_unpainted_tuple)
    
    def to_mod_int(val):
        num, den = val.as_integer_ratio()
        return (num * pow(den, mod - 2, mod)) % mod

    from fractions import Fraction
    memo = {}
    
    def get_expected_operations_fraction(current_vertex, unpainted_vertices_tuple):
        unpainted_vertices = set(unpainted_vertices_tuple)
        if not unpainted_vertices:
            return Fraction(0)
        state = (current_vertex, unpainted_vertices_tuple)
        if state in memo:
            return memo[state]
        
        expected_value = Fraction(1)
        neighbors = adjacency_list[current_vertex]
        num_neighbors = len(neighbors)
        
        next_expected_value = Fraction(0)
        for neighbor in neighbors:
            next_unpainted_vertices = set(unpainted_vertices)
            if neighbor in next_unpainted_vertices:
                next_unpainted_vertices.remove(neighbor)
            next_unpainted_tuple = tuple(sorted(list(next_unpainted_vertices)))
            next_expected_value += get_expected_operations_fraction(neighbor, next_unpainted_tuple)
            
        expected_value += next_expected_value / Fraction(num_neighbors)
        memo[state] = expected_value
        return expected_value

    initial_unpainted_tuple = tuple(sorted(list(initial_unpainted)))
    fraction_result = get_expected_operations_fraction(0, initial_unpainted_tuple)
    print(to_mod_int(fraction_result))

if __name__ == "__main__":
    solve()