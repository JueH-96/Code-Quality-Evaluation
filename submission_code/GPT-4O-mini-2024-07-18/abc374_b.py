# YOUR CODE HERE
def find_difference_position(S, T):
    min_length = min(len(S), len(T))
    
    for i in range(min_length):
        if S[i] != T[i]:
            return i + 1  # +1 for 1-based index
    
    # If we reach here, it means the first min_length characters are the same
    if len(S) != len(T):
        return min_length + 1  # The first differing position is after the last common character
    
    return 0  # They are equal

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
T = data[1]

# Get the result and print it
result = find_difference_position(S, T)
print(result)