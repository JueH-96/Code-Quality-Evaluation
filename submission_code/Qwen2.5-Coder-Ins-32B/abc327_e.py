# YOUR CODE HERE
import sys
import math

def calculate_rating(performances):
    n = len(performances)
    max_rating = float('-inf')
    
    # Precompute the weights
    weights = [(0.9 ** i) for i in range(n)]
    
    # Try all subsets of performances
    for i in range(1, 1 << n):
        total_weighted_performance = 0
        total_weight = 0
        count = 0
        for j in range(n):
            if i & (1 << j):
                total_weighted_performance += weights[count] * performances[j]
                total_weight += weights[count]
                count += 1
        rating = (total_weighted_performance / total_weight) - (1200 / math.sqrt(count))
        max_rating = max(max_rating, rating)
    
    return max_rating

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    performances = list(map(int, input[1:]))
    result = calculate_rating(performances)
    print(f"{result:.15f}")

if __name__ == "__main__":
    main()