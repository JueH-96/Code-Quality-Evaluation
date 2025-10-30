import sys

def solve_test_case():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Sort the array in ascending order
    A.sort()
    
    # Check if the array is already non-decreasing
    if all(A[i] <= A[i+1] for i in range(N-1)):
        return "Yes"
    
    # Try to make the array non-decreasing by performing the given operations
    for i in range(N-1):
        for j in range(i+1, N):
            # Perform the operation (i, j)
            new_A = A[:]
            new_A[i] += 1
            new_A[j] -= 1
            
            # Sort the modified array and check if it is non-decreasing
            new_A.sort()
            if all(new_A[k] <= new_A[k+1] for k in range(N-1)):
                return "Yes"
    
    return "No"

T = int(input())
for _ in range(T):
    print(solve_test_case())