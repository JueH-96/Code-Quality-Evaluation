# YOUR CODE HERE
import sys
from collections import Counter

def solve():
    """
    Reads input, calculates the maximum probability of two chosen dice showing the same number,
    and prints the result.
    """
    # Read the number of dice
    n = int(sys.stdin.readline())

    # Store the number of faces (K_i) and the counts of each face value for each die
    k_values = []
    dice_counters = []
    for _ in range(n):
        # Read the line for the current die: K_i A_{i,1} A_{i,2} ... A_{i,K_i}
        line = list(map(int, sys.stdin.readline().split()))
        
        # Extract K_i (number of faces)
        # Constraint: K_i >= 1
        k = line[0]
        
        # Extract the face values A_{i,j}
        # Constraint: 1 <= A_{i,j} <= 10^5
        faces = line[1:]
        # Assert len(faces) == k # This should hold based on problem statement format
        
        # Store K_i
        k_values.append(k)
        
        # Use collections.Counter to efficiently store the counts of each face value
        # Example: If faces = [1, 2, 2, 1], Counter will be {1: 2, 2: 2}
        dice_counters.append(Counter(faces))

    # Initialize the variable to store the maximum probability found
    max_prob = 0.0

    # Iterate through all unique pairs of distinct dice (i, j)
    # The outer loop runs from i = 0 to N-2
    # The inner loop runs from j = i+1 to N-1
    # This covers all pairs (i, j) where 0 <= i < j < N exactly once.
    for i in range(n):
        for j in range(i + 1, n):
            # Retrieve the pre-computed data for dice i and j
            counts_i = dice_counters[i] # Counter object for die i
            counts_j = dice_counters[j] # Counter object for die j
            k_i = k_values[i]           # Number of faces for die i
            k_j = k_values[j]           # Number of faces for die j

            # Calculate the sum required for the probability numerator.
            # The probability P(X_i = X_j) is given by:
            # Sum_{v} [ P(X_i = v) * P(X_j = v) ]
            # = Sum_{v} [ (count_i(v) / K_i) * (count_j(v) / K_j) ]
            # = (1 / (K_i * K_j)) * Sum_{v} [ count_i(v) * count_j(v) ]
            # We need to compute the term Sum_{v} [ count_i(v) * count_j(v) ]
            
            intersection_sum = 0

            # Optimization: Iterate over the keys of the smaller Counter object.
            # This strategy reduces the number of lookups needed in the potentially larger Counter,
            # improving efficiency when the number of unique face values differs significantly between dice.
            if len(counts_i) <= len(counts_j):
                smaller_counts = counts_i
                larger_counts = counts_j
            else:
                smaller_counts = counts_j
                larger_counts = counts_i

            # Perform the summation by iterating through unique face values present on the 'smaller' die.
            # For each face value 'v' present on the smaller die:
            for value, count_small in smaller_counts.items():
                # Get the count of this same 'value' from the 'larger' die's counter.
                # The .get(value, 0) method returns the count if 'value' exists, or 0 otherwise.
                count_large = larger_counts.get(value, 0)
                
                # If the value exists on both dice (count_large > 0), its contribution to the sum is
                # (count of value on smaller die) * (count of value on larger die).
                # If count_large is 0, the product is 0, so it correctly adds nothing.
                intersection_sum += count_small * count_large

            # Calculate the probability that dice i and j show the same number when rolled.
            # P(match for pair i, j) = intersection_sum / (K_i * K_j)
            # Constraints ensure K_i >= 1 and K_j >= 1, so the denominator K_i * K_j is always positive.
            # No risk of division by zero.
            current_prob = intersection_sum / (k_i * k_j)

            # Update the overall maximum probability found so far if the current pair's probability is higher.
            max_prob = max(max_prob, current_prob)

    # Print the final maximum probability.
    # The output needs to be accurate to 10^-8. Using f-string formatting with
    # sufficient decimal places ensures this requirement is met. '.12f' provides 12 decimal places.
    print(f"{max_prob:.12f}")

# Execute the solve function to run the main logic of the program
solve()

# END YOUR CODE HERE