def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0].strip()
    n = len(s)
    count = 0
    # Iterate over possible i and j indices.
    # Since j - i = k - j, we have k = 2 * j - i.
    for i in range(n):
        # S_i must be 'A'
        if s[i] != 'A':
            continue
        for j in range(i + 1, n):
            # S_j must be 'B'
            if s[j] != 'B':
                continue
            k = 2 * j - i
            # Check k is within range and S_k is 'C'
            if k < n and s[k] == 'C':
                count += 1
    print(count)

if __name__ == '__main__':
    main()