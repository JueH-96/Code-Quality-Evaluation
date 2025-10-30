import sys
import math

def calculate_rating(performance, k):
    """
    Calculate the rating for a given performance and number of contests.
    
    Args:
    performance (list): A list of performances.
    k (int): The number of contests.
    
    Returns:
    float: The calculated rating.
    """
    numerator = sum([performance[i] * (0.9 ** (k - i - 1)) for i in range(k)])
    denominator = sum([0.9 ** (k - i - 1) for i in range(k)])
    rating = numerator / denominator - 1200 / math.sqrt(k)
    return rating

def find_max_rating(performance):
    """
    Find the maximum possible rating by choosing some contests.
    
    Args:
    performance (list): A list of performances.
    
    Returns:
    float: The maximum possible rating.
    """
    n = len(performance)
    max_rating = float('-inf')
    
    # Generate all possible subsets of contests
    for mask in range(1, 2 ** n):
        subset = [performance[i] for i in range(n) if (mask & (1 << i))]
        k = len(subset)
        
        # Calculate the rating for the current subset
        rating = calculate_rating(subset, k)
        
        # Update the maximum rating
        max_rating = max(max_rating, rating)
    
    return max_rating

def main():
    # Read the input from stdin
    n = int(input())
    performance = list(map(int, input().split()))
    
    # Find the maximum possible rating
    max_rating = find_max_rating(performance)
    
    # Print the result to stdout
    print(max_rating)

if __name__ == "__main__":
    main()