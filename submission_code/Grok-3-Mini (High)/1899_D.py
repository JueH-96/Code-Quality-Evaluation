import sys
from collections import Counter

# Read all input and split into a list of strings
data = sys.stdin.read().split()
index = 0
t = int(data[index])
index += 1

# Process each test case
for _ in range(t):
    n = int(data[index])
    index += 1
    # Read the array a
    a_values = list(map(int, data[index:index + n]))
    index += n
    
    # Compute frequency of each exponent
    freq = Counter(a_values)
    
    # Sum of binomial coefficients for all frequencies
    sum_binom = sum((count * (count - 1)) // 2 for count in freq.values())
    
    # Frequency of exponent 1 and 2
    freq1 = freq.get(1, 0)
    freq2 = freq.get(2, 0)
    
    # Cross term for pairs where one has exponent 1 and one has exponent 2
    cross = freq1 * freq2
    
    # Total number of pairs that satisfy the condition
    total = sum_binom + cross
    
    # Output the result for this test case
    print(total)