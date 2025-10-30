import sys
import math

def main():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    
    max_rating = -float('inf')
    
    # Iterate over all possible k from 1 to N
    for k in range(1, N+1):
        # Initialize the numerator and denominator for the weighted sum
        numerator = 0.0
        denominator = 0.0
        # Calculate the weights for the last k contests
        # The weight for the i-th contest (from the end) is (0.9)^(i-1)
        # So for the last contest, weight is 1.0, for the second last, 0.9, etc.
        # We need to choose any k contests, not necessarily the last k
        # So we need to find the best k contests to maximize the weighted sum
        # To maximize the weighted sum, we should choose the k contests with the highest P_i, but considering their positions
        # However, since the weights are fixed for the last k positions, we need to find the best k contests to place in these positions
        # So we can consider all possible combinations of k contests and calculate the weighted sum
        # But with N up to 5000, this is computationally infeasible
        # Instead, we can use dynamic programming to find the best k contests
        # We can precompute the weights for each position in the last k contests
        # The weight for the i-th position in the last k contests is (0.9)^(k-i)
        # So for the first position in the last k contests, weight is (0.9)^(k-1)
        # For the second position, (0.9)^(k-2), ..., for the last position, (0.9)^0 = 1.0
        # We need to assign the P_i to these positions to maximize the weighted sum
        # To maximize the weighted sum, we should assign the largest P_i to the position with the largest weight
        # So we sort the P_i in descending order and assign them to the positions in order of their weights
        # So first, we sort P in descending order
        sorted_P = sorted(P, reverse=True)
        # Now, we assign the first k sorted P_i to the positions with the weights in descending order
        # The weights are (0.9)^(k-1), (0.9)^(k-2), ..., 1.0
        # So the weight for the i-th position is (0.9)^(k-i)
        for i in range(k):
            weight = (0.9) ** (k - i - 1)
            numerator += weight * sorted_P[i]
            denominator += weight
        # Calculate the rating
        rating = (numerator / denominator) - (1200 / math.sqrt(k))
        # Update the maximum rating
        if rating > max_rating:
            max_rating = rating
    
    # Print the maximum rating with 15 decimal places
    print("{0:.15f}".format(max_rating))

if __name__ == "__main__":
    main()