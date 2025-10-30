def solution():
    N, Q = map(int, input().split())
    pattern = [input() for _ in range(N)]
    
    for _ in range(Q):
        A, B, C, D = map(int, input().split())
        black_count = 0
        
        for r in range(N):
            for c in range(N):
                if pattern[r][c] == 'B':
                    # Compute the count of cells with pattern position (r, c)
                    k_start = (A - r + N - 1) // N if A - r > 0 else 0
                    k_end = (C - r) // N
                    
                    l_start = (B - c + N - 1) // N if B - c > 0 else 0
                    l_end = (D - c) // N
                    
                    if k_start <= k_end and l_start <= l_end:
                        black_count += (k_end - k_start + 1) * (l_end - l_start + 1)
        
        print(black_count)

solution()