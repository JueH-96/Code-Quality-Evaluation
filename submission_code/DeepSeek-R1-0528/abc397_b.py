def main():
    S = input().strip()
    n = len(S)
    INF = 10**9
    dp = [[INF] * 2 for _ in range(n+1)]
    dp[0][0] = 0
    max_j = 2 * n + 2

    for total_length in range(0, max_j + 1):
        for i in range(n+1):
            for p in [0, 1]:
                if dp[i][p] != total_length:
                    continue
                if total_length + 1 <= max_j:
                    new_p = 1 - p
                    if total_length + 1 < dp[i][new_p]:
                        dp[i][new_p] = total_length + 1
                if i < n:
                    req_char = 'i' if p == 0 else 'o'
                    if S[i] == req_char:
                        new_p = 1 - p
                        if total_length + 1 < dp[i+1][new_p]:
                            dp[i+1][new_p] = total_length + 1

    if dp[n][1] + 1 < dp[n][0]:
        dp[n][0] = dp[n][1] + 1

    answer = dp[n][0] - n
    print(answer)

if __name__ == "__main__":
    main()