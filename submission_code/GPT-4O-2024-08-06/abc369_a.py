# YOUR CODE HERE
def count_arithmetic_x(A, B):
    possible_x = set()
    
    # Calculate possible x values
    x1 = 2 * B - A
    x2 = (A + B) / 2
    x3 = 2 * A - B
    
    # Add x1 and x3 directly since they are always integers
    possible_x.add(x1)
    possible_x.add(x3)
    
    # Add x2 only if it is an integer
    if x2.is_integer():
        possible_x.add(int(x2))
    
    # Return the number of unique x values
    return len(possible_x)

# Read input
import sys
input = sys.stdin.read
data = input().strip()
A, B = map(int, data.split())

# Output the result
print(count_arithmetic_x(A, B))