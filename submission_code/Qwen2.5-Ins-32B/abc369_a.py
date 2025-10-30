import sys

def count_arithmetic_sequences(A, B):
    # If A and B are the same, only one value of x (which is A or B) can form an arithmetic sequence.
    if A == B:
        return 1
    
    # Calculate the possible values of x that can form an arithmetic sequence with A and B.
    x1 = 2 * A - B  # When B is the middle term
    x2 = 2 * B - A  # When A is the middle term
    x3 = (A + B) // 2  # When x is the middle term, but only if A and B are of the same parity
    
    # Count the unique values of x.
    unique_x = set([x1, x2])
    if (A + B) % 2 == 0:
        unique_x.add(x3)
    
    return len(unique_x)

# Read input from stdin
A, B = map(int, sys.stdin.readline().strip().split())

# Calculate and print the result
print(count_arithmetic_sequences(A, B))