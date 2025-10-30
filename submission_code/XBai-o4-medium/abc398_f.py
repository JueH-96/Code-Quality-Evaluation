def main():
    import sys
    input = sys.stdin.read().strip()
    s = input
    rev_s = s[::-1]
    t = rev_s + '#' + s
    n = len(t)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i-1]
        while j > 0 and t[i] != t[j]:
            j = pi[j-1]
        if t[i] == t[j]:
            j += 1
        pi[i] = j
    max_len = pi[-1]
    result = s + rev_s[max_len:]
    print(result)

if __name__ == "__main__":
    main()