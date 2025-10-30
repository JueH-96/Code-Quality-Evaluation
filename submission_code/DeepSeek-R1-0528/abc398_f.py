def main():
    S = input().strip()
    n = len(S)
    rev = S[::-1]
    pattern = rev + '#' + S
    m = len(pattern)
    pi = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j-1]
        if pattern[i] == pattern[j]:
            j += 1
        else:
            j = 0
        pi[i] = j
        
    k = pi[m-1]
    result = S + rev[k:]
    print(result)

if __name__ == "__main__":
    main()