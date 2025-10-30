def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    n = len(S)
    count = 0
    
    # Iterate over all possible j (middle index)
    for j in range(1, n - 1):
        if S[j] != 'B':
            continue
        
        # Count possible 'A' before j
        count_A = 0
        for i in range(j):
            if S[i] == 'A':
                count_A += 1
        
        # Count possible 'C' after j
        count_C = 0
        for k in range(j + 1, n):
            if S[k] == 'C':
                count_C += 1
        
        # For each 'B' at position j, the number of valid (i, j, k) is count_A * count_C
        count += count_A * count_C
    
    print(count)

if __name__ == "__main__":
    main()