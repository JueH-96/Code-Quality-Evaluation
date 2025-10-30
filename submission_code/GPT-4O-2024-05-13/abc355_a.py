# YOUR CODE HERE
def find_culprit(A, B):
    suspects = {1, 2, 3}
    not_culprits = {A, B}
    
    possible_culprits = suspects - not_culprits
    
    if len(possible_culprits) == 1:
        return possible_culprits.pop()
    else:
        return -1

# Read input
import sys
input = sys.stdin.read().strip()
A, B = map(int, input.split())

# Find and print the culprit
print(find_culprit(A, B))