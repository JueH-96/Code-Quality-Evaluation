# YOUR CODE HERE
def count_valid_triples(N, A):
    from collections import defaultdict

    # Dictionary to store the count of each element
    count = defaultdict(int)
    # Dictionary to store the count of elements seen so far
    seen = defaultdict(int)
    
    # Count the occurrences of each element in the array
    for num in A:
        count[num] += 1
    
    result = 0
    
    # Iterate through the array to find valid triples
    for j in range(N):
        count[A[j]] -= 1  # Decrement the count of the current element
        
        if count[A[j]] == 0:
            del count[A[j]]  # Remove the element if its count becomes zero
        
        if A[j] in seen:
            # If A[j] has been seen before, it cannot be A_i or A_k
            result += seen[A[j]] * count[A[j]]
        
        seen[A[j]] += 1  # Increment the seen count for the current element
    
    return result

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Get the result and print it
result = count_valid_triples(N, A)
print(result)