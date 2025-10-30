# YOUR CODE HERE
import sys

def count_operations(A):
    operations = 0
    while len([x for x in A if x > 0]) > 1:
        A.sort(reverse=True)
        A[0] -= 1
        A[1] -= 1
        operations += 1
    return operations

N = int(input())
A = list(map(int, input().split()))

print(count_operations(A))