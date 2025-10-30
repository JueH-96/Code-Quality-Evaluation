# YOUR CODE HERE
from collections import defaultdict

def count_triples():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Dictionary to store indices for each value
    value_indices = defaultdict(list)
    for i, value in enumerate(A):
        value_indices[value].append(i)
    
    count = 0
    for j in range(1, N-1):
        left = defaultdict(int)
        right = defaultdict(int)
        
        # Count elements to the left of j
        for i in range(j):
            if A[i] != A[j]:
                left[A[i]] += 1
        
        # Count elements to the right of j
        for k in range(j+1, N):
            if A[k] != A[j]:
                right[A[k]] += 1
        
        # Count valid triples
        for value in left:
            count += left[value] * right[value]
    
    print(count)

count_triples()