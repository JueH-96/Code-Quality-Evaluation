def main():
    s = input().strip()
    n = len(s)
    if n == 0:
        print('')
        return
    r = s[::-1]
    T = r + '#' + s
    m = len(T)
    pi = [0] * m
    for i in range(1, m):
        j = pi[i-1]
        while j > 0 and T[i] != T[j]:
            j = pi[j-1]
        if T[i] == T[j]:
            j += 1
        pi[i] = j
    l = pi[-1]
    prefix = s[:n - l]
    ans = s + prefix[::-1]
    print(ans)

if __name__ == "__main__":
    main()