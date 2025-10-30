def insert_numbers(A):
    N = len(A)
    while True:
        # Check if every pair of adjacent terms has an absolute difference of 1
        if all(abs(A[i] - A[i+1]) == 1 for i in range(N-1)):
            break
        
        # Find the pair of adjacent terms nearest to the beginning with absolute difference not 1
        for i in range(N-1):
            if abs(A[i] - A[i+1]) != 1:
                break
        
        # Insert the missing numbers between A[i] and A[i+1]
        if A[i] < A[i+1]:
            A[i+1:i+1] = range(A[i]+1, A[i+1])
        else:
            A[i+1:i+1] = range(A[i]-1, A[i+1], -1)
    
    return A

# Read input
N = int(input())
A = [int(x) for x in input().split()]

# Solve the problem
result = insert_numbers(A)

# Print the result
print(" ".join(map(str, result)))