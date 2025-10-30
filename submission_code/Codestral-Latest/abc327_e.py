import sys
import math

def calculate_rating(performances):
    n = len(performances)
    max_rating = float('-inf')

    for i in range(n):
        for j in range(i, n):
            k = j - i + 1
            numerator = 0
            denominator = 0
            for l in range(k):
                numerator += (0.9 ** (k - l - 1)) * performances[i + l]
                denominator += (0.9 ** (k - l - 1))
            rating = numerator / denominator - 1200 / math.sqrt(k)
            max_rating = max(max_rating, rating)

    return max_rating

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    performances = list(map(int, data[1:N+1]))

    result = calculate_rating(performances)
    print(result)

if __name__ == "__main__":
    main()