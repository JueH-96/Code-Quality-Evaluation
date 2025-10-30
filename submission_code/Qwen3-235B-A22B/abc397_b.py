import sys

def main():
    s = sys.stdin.readline().strip()
    n = len(s)
    INF = float('inf')
    dp = [[INF] * 2 for _ in range(n + 1)]
    dp[0][0] = 0  # Starting at position 0 expecting 'i'

    for i in range(n + 1):
        for state in range(2):
            if dp[i][state] == INF:
                continue
            # Option 1: Insert a character
            new_state_insert = 1 - state
            if dp[i][new_state_insert] > dp[i][state] + 1:
                dp[i][new_state_insert] = dp[i][state] + 1
            # Option 2: Use current character if matches expected
            if i < n:
                expected = 'i' if state == 0 else 'o'
                if s[i] == expected:
                    new_i = i + 1
                    new_state = 1 - state
                    if dp[new_i][new_state] > dp[i][state]:
                        dp[new_i][new_state] = dp[i][state]
    
    # The answer is the minimum insertions required to reach even length
    # If ends in state 1, need to add 1 insertion of 'o'
    result = min(dp[n][0], dp[n][1] + 1 if dp[n][1] != INF else INF)
    print(result)

if __name__ == "__main__":
    main()