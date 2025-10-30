def main():
    S = input().strip()
    n = len(S)
    count = 0
    # We want triples (i, j, k) with i<j<k, j-i = k-j, and S[i]='A', S[j]='B', S[k]='C'.
    # Zero‐based indices: i in [0..n), j in [i+1..n), k = 2*j - i
    for i in range(n):
        if S[i] != 'A':
            continue
        for j in range(i + 1, n):
            if S[j] != 'B':
                continue
            k = 2 * j - i
            if k < n and S[k] == 'C':
                count += 1
    print(count)

if __name__ == "__main__":
    main()