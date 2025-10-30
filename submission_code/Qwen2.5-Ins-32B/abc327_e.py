import sys
from math import sqrt

def calculate_rating(performances):
    n = len(performances)
    dp = [0] * (n + 1)
    for i in range(n):
        dp[i + 1] = dp[i] + performances[i] * (0.9 ** i)
    max_rating = float('-inf')
    for k in range(1, n + 1):
        rating = (dp[k] - dp[0]) / (sum(0.9 ** i for i in range(k))) - 1200 / sqrt(k)
        max_rating = max(max_rating, rating)
    return max_rating

def main():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    performances = list(map(int, data[1:n+1]))
    print(calculate_rating(performances))

if __name__ == "__main__":
    main()