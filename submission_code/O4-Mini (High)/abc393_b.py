def main():
    S = input().strip()
    n = len(S)
    ans = 0
    for i in range(n):
        if S[i] != 'A':
            continue
        for j in range(i+1, n):
            if S[j] != 'B':
                continue
            k = 2*j - i
            if k < n and S[k] == 'C':
                ans += 1
    print(ans)

if __name__ == "__main__":
    main()