def find_A(B):
    # We will check for possible values of A starting from 1 upwards
    A = 1
    while True:
        power = A ** A
        if power == B:
            return A
        elif power > B:
            break
        A += 1
    return -1

# Read input
import sys
input = sys.stdin.read
B = int(input().strip())

# Find and print the result
result = find_A(B)
print(result)