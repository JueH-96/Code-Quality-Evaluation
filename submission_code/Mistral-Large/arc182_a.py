import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    Q = int(data[1])
    P = [0] * Q
    V = [0] * Q

    for i in range(Q):
        P[i] = int(data[2 * i + 2])
        V[i] = int(data[2 * i + 3])

    # Initialize dp arrays
    dp_left = [[0] * (N + 1) for _ in range(Q + 1)]
    dp_right = [[0] * (N + 1) for _ in range(Q + 1)]

    # Base case
    for j in range(1, N + 1):
        dp_left[0][j] = 1
        dp_right[0][j] = 1

    # Fill dp arrays
    for i in range(1, Q + 1):
        for j in range(1, N + 1):
            dp_left[i][j] = dp_left[i][j - 1]
            dp_right[i][j] = dp_right[i][j - 1]
            if j <= P[i - 1] and V[i - 1] >= j:
                dp_left[i][j] += dp_left[i - 1][j]
                dp_left[i][j] %= MOD
            if j >= P[i - 1] and V[i - 1] >= j:
                dp_right[i][j] += dp_right[i - 1][j]
                dp_right[i][j] %= MOD

    # Calculate the result
    result = 0
    for j in range(1, N + 1):
        result += dp_left[Q][j] * dp_right[Q][j]
        result %= MOD

    print(result)

if __name__ == "__main__":
    main()