def main():
    import sys
    A, M, L, R = map(int, sys.stdin.read().split())
    
    # Compute k_min = ceil((L - A)/M)
    if (L - A) % M == 0:
        k_min = (L - A) // M
    else:
        k_min = (L - A) // M + 1 if (L - A) > 0 else (L - A) // M
    
    # Compute k_max = floor((R - A)/M)
    k_max = (R - A) // M
    
    # Adjust k_min and k_max to ensure positions are within [L, R]
    pos_k_min = A + k_min * M
    if pos_k_min < L:
        k_min += 1
    
    pos_k_max = A + k_max * M
    if pos_k_max > R:
        k_max -= 1
    
    # Calculate the number of k's
    if k_max >= k_min:
        print(k_max - k_min + 1)
    else:
        print(0)

if __name__ == '__main__':
    main()