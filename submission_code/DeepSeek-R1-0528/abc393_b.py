def main():
    S = input().strip()
    n = len(S)
    count = 0
    for j in range(1, n-1):
        if S[j] == 'B':
            max_d = min(j, n-1 - j)
            for d in range(1, max_d + 1):
                if S[j - d] == 'A' and S[j + d] == 'C':
                    count += 1
    print(count)

if __name__ == "__main__":
    main()