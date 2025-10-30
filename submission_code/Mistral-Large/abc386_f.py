import sys

def can_transform(K, S, T):
    # If the lengths of S and T differ by more than K, it's impossible to transform S to T
    if abs(len(S) - len(T)) > K:
        return "No"

    # Calculate the minimum number of operations needed to transform S to T
    n = len(S)
    m = len(T)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif S[i - 1] == T[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return "Yes" if dp[n][m] <= K else "No"

def main():
    input = sys.stdin.read
    data = input().split()

    K = int(data[0])
    S = data[1]
    T = data[2]

    result = can_transform(K, S, T)
    print(result)

if __name__ == "__main__":
    main()