def main():
    import sys
    S = sys.stdin.readline().strip()
    if not S:
        print(S)
        return
    rev_S = S[::-1]
    T = rev_S + '#' + S
    n = len(T)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i-1]
        while j > 0 and T[i] != T[j]:
            j = pi[j-1]
        if T[i] == T[j]:
            j += 1
        pi[i] = j
    k = pi[-1]
    add_part = S[:len(S) - k]
    result = S + add_part[::-1]
    print(result)

if __name__ == "__main__":
    main()