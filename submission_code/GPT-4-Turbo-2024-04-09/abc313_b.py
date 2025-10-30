def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    from collections import defaultdict, deque
    
    # Graph adjacency list
    stronger_than = defaultdict(list)
    weaker_than = defaultdict(list)
    
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        index += 2
        stronger_than[A].append(B)
        weaker_than[B].append(A)
    
    # Use Floyd-Warshall-like approach to determine all relations
    can_beat = [[False] * (N + 1) for _ in range(N + 1)]
    
    # Initialize direct relations
    for i in range(1, N + 1):
        for j in stronger_than[i]:
            can_beat[i][j] = True
    
    # Transitive closure using a modified Floyd-Warshall
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if can_beat[i][k] and can_beat[k][j]:
                    can_beat[i][j] = True
    
    # Determine the strongest programmer
    strongest_candidate = -1
    
    for i in range(1, N + 1):
        is_strongest = True
        for j in range(1, N + 1):
            if i != j and not can_beat[i][j]:
                is_strongest = False
                break
        if is_strongest:
            if strongest_candidate != -1:
                # More than one strongest candidate found
                print(-1)
                return
            strongest_candidate = i
    
    print(strongest_candidate)

if __name__ == "__main__":
    main()