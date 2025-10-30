# YOUR CODE HERE
import sys
from collections import Counter

# Read N
N = int(sys.stdin.readline())

# Read dice data
# Store as list of tuples: (K_i, Counter_i)
# K_i is the total number of faces.
# Counter_i maps value to frequency on die i.
dice_data = []
for _ in range(N):
    line = list(map(int, sys.stdin.readline().split()))
    K = line[0]
    values = line[1:]
    counts = Counter(values)
    dice_data.append((K, counts))

# Calculate maximum probability
max_prob = 0.0

# Iterate through all distinct pairs of dice (i, j) where i < j
for i in range(N):
    for j in range(i + 1, N):
        # Get data for die i and die j
        K_i, counts_i = dice_data[i]
        K_j, counts_j = dice_data[j]

        current_prob = 0.0
        
        # The probability that die i and die j show the same number is
        # Sum over all possible values v: P(die i shows v AND die j shows v)
        # Since rolls are independent:
        # Sum over all possible values v: P(die i shows v) * P(die j shows v)
        # P(die i shows v) = count_i(v) / K_i
        # P(die j shows v) = count_j(v) / K_j
        # Sum = Sum_v [ (count_i(v) / K_i) * (count_j(v) / K_j) ]
        
        # The sum only gets non-zero contributions from values v that appear
        # on *both* die i and die j (i.e., v is in the intersection of their face values).
        # We can iterate through the unique values present in the die with fewer
        # unique values and check if they are present in the other die.
        
        # Determine which counter has fewer unique keys to iterate over
        if len(counts_i) <= len(counts_j):
            smaller_counts = counts_i
            larger_counts = counts_j
            K_smaller = K_i
            K_larger = K_j
        else:
            smaller_counts = counts_j
            larger_counts = counts_i
            K_smaller = K_j
            K_larger = K_i

        # Use float division to ensure correct probability calculation
        K_smaller_f = float(K_smaller)
        K_larger_f = float(K_larger)

        # Iterate through the items (value, count) in the Counter with fewer unique keys
        for v, count_smaller_v in smaller_counts.items():
            # Check if this value v also exists as a key in the other Counter
            if v in larger_counts:
                # Get the count of value v from the larger Counter
                count_larger_v = larger_counts[v]
                
                # Add the contribution for value v to the total probability for this pair of dice
                # Contribution = (Probability of v on smaller die) * (Probability of v on larger die)
                contribution = (count_smaller_v / K_smaller_f) * (count_larger_v / K_larger_f)
                current_prob += contribution

        # Update the maximum probability found so far across all pairs
        max_prob = max(max_prob, current_prob)

# Print the result with the required precision (at least 10^{-8})
# Printing with 15 decimal places is usually sufficient for double precision
print(f"{max_prob:.15f}")