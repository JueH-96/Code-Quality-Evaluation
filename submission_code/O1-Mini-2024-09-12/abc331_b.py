import sys

def main():
    N, S, M, L = map(int, sys.stdin.read().split())
    max_eggs = N + 12  # Maximum eggs to consider
    dp = [float('inf')] * (max_eggs + 1)
    dp[0] = 0

    for i in range(max_eggs + 1):
        if dp[i] != float('inf'):
            if i + 6 <= max_eggs:
                dp[i + 6] = min(dp[i + 6], dp[i] + S)
            if i + 8 <= max_eggs:
                dp[i + 8] = min(dp[i + 8], dp[i] + M)
            if i + 12 <= max_eggs:
                dp[i + 12] = min(dp[i + 12], dp[i] + L)

    min_cost = min(dp[N:])
    print(min_cost)

if __name__ == "__main__":
    main()