A, B, C, D = map(int, input().split())

def count_black(A, B, C, D):
    # Calculate the number of x-tiles (n)
    # x ranges from A to C (inclusive)
    # Each x-tile is [n, n+1)
    # Overlap with [A, C] => n <= C and n+1 > A
    # n_min = floor(A)
    # n_max = floor(C - 1e-9) => floor(C - 1) if C is integer
    n_min = A if A >=0 else (A // 1)
    n_max = C - 1 if C > 0 else (C // 1) - 1
    if n_max < n_min:
        n_count = 0
    else:
        n_count = n_max - n_min + 1
    
    # Calculate the number of y-tiles (m)
    # y ranges from B to D (inclusive)
    # Each y-tile is [2m, 2m+2)
    # Overlap with [B, D] => 2m <= D and 2m+2 > B
    # m_min = ceil( (B - 2) / 2 )
    # m_max = floor( D / 2 )
    m_min = (B - 2 + 1) // 2
    m_max = D // 2
    if m_max < m_min:
        m_count = 0
    else:
        m_count = m_max - m_min + 1
    
    # Now calculate the area contribution
    total = 0
    # Handle full tiles
    full_n = max(0, n_max - n_min + 1)
    full_m = max(0, m_max - m_min + 1)
    if full_n > 0 and full_m > 0:
        total_full = full_n * full_m
        if ( (n_min + m_min) % 2 == 0 ):
            black = (total_full + 1) // 2
        else:
            black = total_full // 2
        total += black
    
    # Handle edge cases where tiles are partially inside
    # Left edge (x = A)
    if A < n_min:
        for n in [n_min - 1]:
            if n < A:
                continue
            if n + 1 <= C:
                y_min_m = m_min
                y_max_m = m_max
                width = min(n + 1 - A, 1)
                for m in range(y_min_m, y_max_m + 1):
                    if 2 * m >= B and 2 * m + 2 <= D:
                        if (n + m) % 2 == 0:
                            total += width * (2 * (2 * m + 2 - D) if 2 * m + 2 > D else 1) if 2 * m + 2 > D else width * 2
                    else:
                        y_overlap = min(2 * m + 2, D) - max(2 * m, B)
                        if y_overlap > 0:
                            total += width * y_overlap
    # Right edge (x = C)
    if C > n_max:
        for n in [n_max + 1]:
            if n >= C:
                continue
            if n + 1 > C:
                width = n + 1 - C
            else:
                continue
            y_min_m = m_min
            y_max_m = m_max
            for m in range(y_min_m, y_max_m + 1):
                if 2 * m >= B and 2 * m + 2 <= D:
                    if (n + m) % 2 == 0:
                        total += width * (2 * (2 * m + 2 - D) if 2 * m + 2 > D else 1) if 2 * m + 2 > D else width * 2
                else:
                    y_overlap = min(2 * m + 2, D) - max(2 * m, B)
                    if y_overlap > 0:
                        total += width * y_overlap
    # Bottom edge (y = B)
    if B < 2 * m_min:
        for m in [m_min - 1]:
            if 2 * m + 2 <= D:
                height = min(2 * m + 2 - B, 2)
                for n in range(n_min, n_max + 1):
                    if (n + m) % 2 == 0:
                        total += height * (n + 1 - max(A, n))
                    else:
                        pass
    # Top edge (y = D)
    if D > 2 * m_max:
        for m in [m_max + 1]:
            if 2 * m >= B:
                height = min(2 * m - D, 2)
                for n in range(n_min, n_max + 1):
                    if (n + m) % 2 == 0:
                        total += height * (n + 1 - max(A, n))
    return total

area = count_black(A, B, C, D)
print(2 * area)