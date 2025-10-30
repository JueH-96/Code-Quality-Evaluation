# YOUR CODE HERE
import math
from collections import Counter

def count_equal_combinations(a):
    n = len(a)
    if n < 2:
        return 0
    
    # Count the frequency of each element
    freq = Counter(a)
    
    # Function to check if two numbers have the same combination
    def is_equal_combination(x, y):
        return math.gcd(x, y) > 1
    
    # Count pairs (i, j) such that b_i^{b_j} = b_j^{b_i}
    count = 0
    for x in freq:
        for y in freq:
            if x < y and is_equal_combination(x, y):
                count += freq[x] * freq[y]
            elif x == y and freq[x] > 1:
                count += freq[x] * (freq[x] - 1) // 2
    
    return count

# Read input
import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
results = []

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n
    results.append(count_equal_combinations(a))

# Write output
for result in results:
    print(result)