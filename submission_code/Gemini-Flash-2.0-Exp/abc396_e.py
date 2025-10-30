def solve():
    n, m = map(int, input().split())
    constraints = []
    for _ in range(m):
        x, y, z = map(int, input().split())
        constraints.append((x - 1, y - 1, z))

    def check_sequence(a):
        for x, y, z in constraints:
            if (a[x] ^ a[y]) != z:
                return False
        return True

    def find_min_sum_sequence():
        import itertools
        
        for a in itertools.product(range(256), repeat=n):
            if check_sequence(a):
                return list(a)
        return None

    
    
    parent = list(range(n))
    xor_values = [0] * n

    def find(i):
        if parent[i] == i:
            return i
        root = find(parent[i])
        xor_values[i] ^= xor_values[parent[i]]
        parent[i] = root
        return root

    def union(i, j, z):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            xor_values[root_j] = xor_values[i] ^ xor_values[j] ^ z
            return True
        else:
            return (xor_values[i] ^ xor_values[j]) == z

    possible = True
    for x, y, z in constraints:
        if not union(x, y, z):
            possible = False
            break

    if not possible:
        print("-1")
        return

    result = [0] * n
    for i in range(n):
        find(i)
        result[i] = xor_values[i]
    
    print(*result)

solve()