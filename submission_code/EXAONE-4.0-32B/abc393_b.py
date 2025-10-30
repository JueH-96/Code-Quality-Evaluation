def main():
    S = input().strip()
    n = len(S)
    count = 0
    for j in range(1, n-1):
        max_d = min(j, n-1-j)
        for d in range(1, max_d+1):
            i_index = j - d
            k_index = j + d
            if S[i_index] == 'A' and S[j] == 'B' and S[k_index] == 'C':
                count += 1
    print(count)

if __name__ == "__main__":
    main()