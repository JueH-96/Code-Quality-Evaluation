# YOUR CODE HERE
def count_valid_triples(N, A):
    from collections import defaultdict
    
    # Dictionary to keep track of counts of each number after a given index
    count_after = defaultdict(int)
    
    # Populate count_after by traversing from right to left
    for num in A:
        count_after[num] += 1
    
    # Dictionary to keep track of counts of each number before a given index
    count_before = defaultdict(int)
    
    # Result variable to store the number of valid triples
    result = 0
    
    # Traverse the array from left to right
    for j in range(N):
        # Current number
        current = A[j]
        
        # Decrease the count of current number in count_after
        count_after[current] -= 1
        
        # Calculate the number of valid triples with j as the middle element
        # We need to count all pairs (i, k) such that A[i] = A[k] = current and A[i] != A[j]
        # This is equivalent to count_before[current] * count_after[current]
        result += count_before[current] * count_after[current]
        
        # Increase the count of current number in count_before
        count_before[current] += 1
    
    return result

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Output the result
print(count_valid_triples(N, A))