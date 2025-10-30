def can_convert_with_k_operations(K, S, T):
    len_s = len(S)
    len_t = len(T)

    # If the difference in lengths is greater than K, it's impossible
    if abs(len_s - len_t) > K:
        return "No"

    # Calculate the edit distance using dynamic programming
    dp = [[0] * (len_t + 1) for _ in range(len_s + 1)]

    for i in range(len_s + 1):
        dp[i][0] = i  # Deleting all characters from S
    for j in range(len_t + 1):
        dp[0][j] = j  # Inserting all characters to form T

    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            if S[i - 1] == T[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    # Deletion
                    dp[i][j - 1] + 1,    # Insertion
                    dp[i - 1][j - 1] + 1  # Replacement
                )

    edit_distance = dp[len_s][len_t]

    # Check if the edit distance is within the allowed number of operations
    return "Yes" if edit_distance <= K else "No"

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    K = int(data[0])
    S = data[1]
    T = data[2]
    
    result = can_convert_with_k_operations(K, S, T)
    print(result)

if __name__ == "__main__":
    main()