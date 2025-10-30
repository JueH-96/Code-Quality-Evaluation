import sys

def calculate_min_time(X, Y, Z, S):
    """
    Calculate the minimum time required to type the string S using the given keyboard.

    Args:
    X (int): The time required to press the 'a' key.
    Y (int): The time required to press the 'a' key and Shift key simultaneously.
    Z (int): The time required to press the Caps Lock key.
    S (str): The string to be typed.

    Returns:
    int: The minimum time required to type the string S.
    """
    n = len(S)
    dp = [[float('inf')] * 2 for _ in range(n + 1)]
    dp[0][0] = 0
    dp[0][1] = Z

    for i in range(1, n + 1):
        if S[i - 1].islower():
            dp[i][0] = min(dp[i - 1][0] + X, dp[i - 1][1] + Y)
            dp[i][1] = min(dp[i - 1][0] + Y + Z, dp[i - 1][1] + X + Z)
        else:
            dp[i][0] = min(dp[i - 1][0] + Y, dp[i - 1][1] + X)
            dp[i][1] = min(dp[i - 1][0] + X + Z, dp[i - 1][1] + Y + Z)

    return min(dp[n][0], dp[n][1])

def main():
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    min_time = calculate_min_time(X, Y, Z, S)
    print(min_time)

if __name__ == "__main__":
    main()