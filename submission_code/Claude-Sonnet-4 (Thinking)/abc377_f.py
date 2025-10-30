def solve():
    N, M = map(int, input().split())
    pieces = []
    for _ in range(M):
        a, b = map(int, input().split())
        pieces.append((a, b))
    
    forbidden_rows = set()
    forbidden_cols = set()
    forbidden_diag1 = set()  # i + j values
    forbidden_diag2 = set()  # i - j values
    
    for a, b in pieces:
        forbidden_rows.add(a)
        forbidden_cols.add(b)
        forbidden_diag1.add(a + b)
        forbidden_diag2.add(a - b)
    
    def diag1_count(s):
        # Count squares (i, j) where i + j = s, 1 <= i <= N, 1 <= j <= N
        if s < 2 or s > 2 * N:
            return 0
        return max(0, min(N, s - 1) - max(1, s - N) + 1)
    
    def diag2_count(d):
        # Count squares (i, j) where i - j = d, 1 <= i <= N, 1 <= j <= N
        if d < 1 - N or d > N - 1:
            return 0
        return max(0, min(N, N + d) - max(1, d + 1) + 1)
    
    # Count attacked squares using inclusion-exclusion
    
    # Single sets
    A1 = len(forbidden_rows) * N
    A2 = len(forbidden_cols) * N
    A3 = sum(diag1_count(s) for s in forbidden_diag1)
    A4 = sum(diag2_count(d) for d in forbidden_diag2)
    
    # Pairwise intersections
    A12 = len(forbidden_rows) * len(forbidden_cols)
    
    A13 = 0
    for r in forbidden_rows:
        for s in forbidden_diag1:
            j = s - r
            if 1 <= j <= N:
                A13 += 1
    
    A14 = 0
    for r in forbidden_rows:
        for d in forbidden_diag2:
            j = r - d
            if 1 <= j <= N:
                A14 += 1
    
    A23 = 0
    for c in forbidden_cols:
        for s in forbidden_diag1:
            i = s - c
            if 1 <= i <= N:
                A23 += 1
    
    A24 = 0
    for c in forbidden_cols:
        for d in forbidden_diag2:
            i = c + d
            if 1 <= i <= N:
                A24 += 1
    
    A34 = 0
    for s in forbidden_diag1:
        for d in forbidden_diag2:
            if (s + d) % 2 == 0:
                i = (s + d) // 2
                j = (s - d) // 2
                if 1 <= i <= N and 1 <= j <= N:
                    A34 += 1
    
    # Triple intersections
    A123 = 0
    for r in forbidden_rows:
        for c in forbidden_cols:
            s = r + c
            if s in forbidden_diag1:
                A123 += 1
    
    A124 = 0
    for r in forbidden_rows:
        for c in forbidden_cols:
            d = r - c
            if d in forbidden_diag2:
                A124 += 1
    
    A134 = 0
    for r in forbidden_rows:
        for s in forbidden_diag1:
            for d in forbidden_diag2:
                if (s + d) % 2 == 0:
                    i = (s + d) // 2
                    j = (s - d) // 2
                    if i == r and 1 <= j <= N:
                        A134 += 1
    
    A234 = 0
    for c in forbidden_cols:
        for s in forbidden_diag1:
            for d in forbidden_diag2:
                if (s + d) % 2 == 0:
                    i = (s + d) // 2
                    j = (s - d) // 2
                    if j == c and 1 <= i <= N:
                        A234 += 1
    
    # Quadruple intersection
    A1234 = 0
    for r in forbidden_rows:
        for c in forbidden_cols:
            s = r + c
            d = r - c
            if s in forbidden_diag1 and d in forbidden_diag2:
                A1234 += 1
    
    # Inclusion-exclusion
    total_attacked = A1 + A2 + A3 + A4 - A12 - A13 - A14 - A23 - A24 - A34 + A123 + A124 + A134 + A234 - A1234
    
    safe_squares = N * N - total_attacked
    
    return safe_squares

print(solve())