def sushi_eating():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    results = []
    
    for sushi in B:
        eaten = -1
        for i in range(N):
            if sushi >= A[i]:
                eaten = i + 1  # +1 because we want 1-based index
                break
        results.append(eaten)
    
    print("
".join(map(str, results)))