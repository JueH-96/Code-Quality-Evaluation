import sys
from math import sqrt

# Read the number of contests
N = int(input().strip())

# Read the performances
performances = list(map(int, input().strip().split()))

# Function to calculate the rating
def calculate_rating(performances):
    k = len(performances)
    numerator = sum((0.9)**(k-i-1) * performances[i] for i in range(k))
    denominator = sum((0.9)**(k-i-1) for i in range(k))
    rating = numerator / denominator - 1200 / sqrt(k)
    return rating

# Initialize the maximum rating
max_rating = float('-inf')

# Check all possible combinations of contests
for i in range(1, 1 << N):
    selected_performances = [performances[j] for j in range(N) if (i >> j) & 1]
    rating = calculate_rating(selected_performances)
    max_rating = max(max_rating, rating)

# Output the maximum possible rating
print(f"{max_rating:.6f}")