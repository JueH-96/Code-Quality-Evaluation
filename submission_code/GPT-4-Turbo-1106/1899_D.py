from collections import Counter
from math import gcd

def count_pairs(a):
    # Count the frequency of each element in the array
    freq = Counter(a)
    # Sort the array to ensure i < j
    sorted_keys = sorted(freq.keys())
    count = 0
    
    # Iterate through each unique pair of notes
    for i in range(len(sorted_keys)):
        for j in range(i + 1, len(sorted_keys)):
            # Check if the pair satisfies the condition b_i^b_j = b_j^b_i
            if gcd(sorted_keys[i], sorted_keys[j]) == 1:
                # If the condition is satisfied, add the product of their frequencies to the count
                count += freq[sorted_keys[i]] * freq[sorted_keys[j]]
    
    # Add the combinations of identical pairs (since a^a = a^a)
    for k in freq:
        if freq[k] > 1:
            count += freq[k] * (freq[k] - 1) // 2
    
    return count

# Read number of test cases
t = int(input().strip())

# Process each test case
for _ in range(t):
    # Read the length of the array
    n = int(input().strip())
    # Read the array elements
    a = list(map(int, input().strip().split()))
    # Output the number of pairs that satisfy the condition
    print(count_pairs(a))