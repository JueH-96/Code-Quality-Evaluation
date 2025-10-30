# YOUR CODE HERE
import math

def find_a(B):
    if B == 1:
        return 1
    for A in range(1, 60):
        current = pow(A, A)
        if current == B:
            return A
        if current > B:
            break
    return -1

B = int(input())
print(find_a(B))