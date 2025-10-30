# YOUR CODE HERE
import sys
import math

def calculate_maximum_rating(N, performances):
    max_rating = float('-inf')
    
    for start in range(N):
        sum_weighted_performance = 0.0
        sum_weights = 0.0
        
        for end in range(start, N):
            k = end - start + 1
            weight = (0.9) ** (k - 1)
            sum_weighted_performance += weight * performances[end]
            sum_weights += weight
            
            rating = (sum_weighted_performance / sum_weights) - (1200 / math.sqrt(k))
            max_rating = max(max_rating, rating)
    
    return max_rating

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    performances = list(map(int, data[1:]))
    
    result = calculate_maximum_rating(N, performances)
    print(f"{result:.15f}")

if __name__ == "__main__":
    main()