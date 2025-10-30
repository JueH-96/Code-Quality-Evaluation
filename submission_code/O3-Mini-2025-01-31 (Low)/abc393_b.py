def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    count = 0
    # Loop through every possible starting index for A (use 0-based index for S)
    for i in range(n-2):
        if S[i] != 'A':
            continue
        # j = i + d and k = i + 2*d, ensure these are within bounds
        for d in range(1, (n - i) // 2 + 1):
            j = i + d
            k = i + 2*d
            if j < n and k < n:
                if S[j] == 'B' and S[k] == 'C':
                    count += 1
    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()