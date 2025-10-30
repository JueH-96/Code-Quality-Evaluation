# Read N, the number of seats
N = int(input())

# Read S, the string representing the state of the seats
S = input()

# Initialize a counter for the number of times the condition is satisfied
count = 0

# The condition requires checking three adjacent seats: i, i+1, and i+2.
# In 0-based indexing for the string S, these correspond to S[k], S[k+1], and S[k+2],
# where k = i-1.
# The problem states that i is between 1 and N-2, inclusive.
# This means k (i-1) will range from (1-1) = 0 up to (N-2-1) = N-3.
# So, we need to iterate k from 0 to N-3.
# In Python, `range(N-2)` generates numbers from 0 up to N-3 (inclusive).

# We only need to iterate if N is at least 3, otherwise, it's impossible to
# find a sequence of three seats. The `range(N-2)` will handle this naturally
# (e.g., if N=1 or N=2, range will be empty), but an explicit check can be clearer.
if N >= 3:
    for k in range(N - 2):
        # Check the condition:
        # S[k] (seat i) is occupied ('#')
        # S[k+1] (seat i+1) is unoccupied ('.')
        # S[k+2] (seat i+2) is occupied ('#')
        if S[k] == '#' and S[k+1] == '.' and S[k+2] == '#':
            count += 1

# Print the final count
print(count)