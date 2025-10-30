# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def count_possible_sets(N, coordinates):
    # Sort coordinates by X, then by Y
    coordinates.sort()
    
    # dp[i] will store the number of valid sets ending at i
    dp = [1] * N
    
    for i in range(N):
        for j in range(i):
            if coordinates[j][1] < coordinates[i][1]:
                dp[i] = (dp[i] + dp[j]) % MOD
    
    return sum(dp) % MOD

def main():
    data = input().split()
    N = int(data[0])
    coordinates = []
    for i in range(N):
        x = int(data[2 * i + 1])
        y = int(data[2 * i + 2])
        coordinates.append((x, y))
    
    result = count_possible_sets(N, coordinates)
    print(result)

if __name__ == "__main__":
    main()