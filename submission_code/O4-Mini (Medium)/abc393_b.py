def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    count = 0
    # We look for triples (i, j, k) with j - i = k - j
    # and S[i] = 'A', S[j] = 'B', S[k] = 'C'.
    # Equivalently, for each i < k with even distance, let j = (i + k)//2.
    for i in range(n):
        if S[i] != 'A':
            continue
        for k in range(i + 2, n, 1):
            if S[k] != 'C':
                continue
            # check if (k - i) is even so j is integer midpoint
            if (k - i) % 2 != 0:
                continue
            j = (i + k) // 2
            if S[j] == 'B':
                count += 1
    print(count)

# Call main to execute
main()