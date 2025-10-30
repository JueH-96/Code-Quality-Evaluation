# YOUR CODE HERE
import sys

def find_A(B):
    if B == 1:
        return 1
    A = 1
    while A ** A <= B:
        if A ** A == B:
            return A
        A += 1
    return -1

B = int(sys.stdin.read().strip())
print(find_A(B))