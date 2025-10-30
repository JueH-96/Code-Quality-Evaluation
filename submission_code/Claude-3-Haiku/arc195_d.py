import sys

def solve_case():
    N = int(input())
    A = list(map(int, input().split()))
    
    operations = 0
    while A:
        # Find the length of the longest prefix of equal elements
        i = 0
        while i < len(A) and A[i] == A[0]:
            i += 1
        
        # If the entire sequence is equal, delete it
        if i == len(A):
            A.clear()
            operations += 1
        # Otherwise, swap adjacent elements until the prefix is deleted
        else:
            operations += len(A) - i
            A = A[i:]
    
    return operations

T = int(input())
for _ in range(T):
    print(solve_case())