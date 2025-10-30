# YOUR CODE HERE
import sys

def min_operations_to_balance(A):
    N = len(A)
    total_sum = sum(A)
    avg = total_sum // N
    operations = 0
    
    for a in A:
        if a > avg:
            operations += a - avg
    
    return operations

N = int(input())
A = list(map(int, input().split()))

print(min_operations_to_balance(A))