def solve():
    n, m = map(int, input().split())
    num_vertices = n * m + 1
    edges = []
    for i in range(1, num_vertices):
        edges.append((i, max(i - n, 0)))
    
    adjacency_list = [[] for _ in range(num_vertices)]
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        
    memo = {}
    
    def get_expected_operations(current_vertex, unpainted_vertices_tuple):
        unpainted_vertices = tuple(sorted(unpainted_vertices_tuple))
        if not unpainted_vertices:
            return 0
        state = (current_vertex, unpainted_vertices)
        if state in memo:
            return memo[state]
        
        expected_value = 1
        neighbors = adjacency_list[current_vertex]
        neighbor_expectation = 0
        
        for neighbor in neighbors:
            next_unpainted_vertices = list(unpainted_vertices)
            if neighbor in unpainted_vertices:
                next_unpainted_vertices.remove(neighbor)
            neighbor_expectation += get_expected_operations(neighbor, tuple(next_unpainted_vertices))
            
        expected_value += neighbor_expectation / len(neighbors)
        memo[state] = expected_value
        return expected_value
        
    initial_unpainted_vertices = tuple(range(1, num_vertices))
    result = get_expected_operations(0, initial_unpainted_vertices)
    
    mod = 998244353
    
    def to_mod_fraction(value):
        from fractions import Fraction
        fraction_value = Fraction(value).limit_denominator()
        numerator = fraction_value.numerator
        denominator = fraction_value.denominator
        
        def power(a, b):
            res = 1
            a %= mod
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % mod
                a = (a * a) % mod
                b //= 2
            return res
            
        denominator_inv = power(denominator, mod - 2)
        return (numerator * denominator_inv) % mod

    print(to_mod_fraction(result))

if __name__ == "__main__":
    solve()