def main():
    import sys
    from itertools import combinations

    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    adjacency = [set() for _ in range(N + 1)]
    existing_edges = set()
    
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        adjacency[A].add(B)
        adjacency[B].add(A)
        existing_edges.add((min(A, B), max(A, B)))
        index += 2
    
    operations = set()
    
    for Y in range(1, N + 1):
        neighbors = list(adjacency[Y])
        for X, Z in combinations(neighbors, 2):
            if (min(X, Z), max(X, Z)) not in existing_edges:
                operations.add((min(X, Z), max(X, Z)))
    
    print(len(operations))

if __name__ == '__main__':
    main()