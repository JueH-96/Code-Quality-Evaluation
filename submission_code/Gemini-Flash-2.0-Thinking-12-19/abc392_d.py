import collections
import sys

# Read N
N = int(sys.stdin.readline())

Ks = []
die_counts = []

# Read dice information
for i in range(N):
    line = list(map(int, sys.stdin.readline().split()))
    K = line[0]
    faces = line[1:]
    Ks.append(K)
    # Use Counter to easily count face occurrences
    counts = collections.Counter(faces)
    die_counts.append(counts)

max_probability = 0.0

# Iterate through all pairs of distinct dice (i, j)
# Use i ranging from 0 to N-1 and j ranging from i+1 to N-1 to avoid duplicate pairs and self-pairing
for i in range(N):
    for j in range(i + 1, N):
        # Calculate the sum of products of counts for matching face values
        match_sum = 0
        
        # To optimize, iterate through the unique values of the die with fewer unique faces
        counts_i = die_counts[i]
        counts_j = die_counts[j]

        # Choose the smaller counter to iterate over its keys
        if len(counts_i) <= len(counts_j):
             smaller_counts = counts_i
             larger_counts = counts_j
        else:
             smaller_counts = counts_j
             larger_counts = counts_i

        # Sum up count_i * count_j for values present in both dice
        for value, count_smaller in smaller_counts.items():
            # Check if the value from the smaller counter is present in the larger counter
            if value in larger_counts:
                count_larger = larger_counts[value]
                match_sum += count_smaller * count_larger

        # The probability of the two dice showing the same number is
        # sum over all matching values x: P(die i shows x) * P(die j shows x)
        # P(die i shows x) = count_i(x) / K_i
        # P(die j shows x) = count_j(x) / K_j
        # Total Probability = sum over all matching values x: (count_i(x) / K_i) * (count_j(x) / K_j)
        # This simplifies to (sum over all matching values x: count_i(x) * count_j(x)) / (K_i * K_j)
        prob = match_sum / (Ks[i] * Ks[j])

        # Update the maximum probability found so far
        max_probability = max(max_probability, prob)

# Print the result with required precision (at least 10^-8 error tolerance)
# 15 decimal places is usually sufficient for standard double precision floats
sys.stdout.write(f"{max_probability:.15f}
")