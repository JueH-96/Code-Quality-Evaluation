def solve():
    A, M, L, R = map(int, input().split())
    
    # Find first tree position >= L
    # (L-A)/M rounded up gives k where A + kM is first position >= L
    k_left = (L - A + M - 1) // M
    first_pos = A + k_left * M
    if first_pos < L:
        first_pos += M
        k_left += 1
        
    # Find last tree position <= R
    # (R-A)/M rounded down gives k where A + kM is last position <= R
    k_right = (R - A) // M
    
    # Count trees between first_pos and last_pos inclusive
    if A + k_left * M > R:
        print(0)
    else:
        count = k_right - k_left + 1
        print(count)

solve()