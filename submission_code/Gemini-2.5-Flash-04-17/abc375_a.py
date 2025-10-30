# YOUR CODE HERE
# Read N
N = int(input())
# Read S
S = input()

# Initialize counter
count = 0

# Iterate through possible starting 0-based indices `j` for the pattern '#.#'.
# The pattern '#.#' corresponds to seats i, i+1, i+2 being occupied, unoccupied, occupied.
# Seat i corresponds to index i-1 in the 0-based string S.
# So we are checking S[i-1], S[i], S[i+1].
# The problem asks for i from 1 to N-2.
# If i=1, we check indices 0, 1, 2. This corresponds to j=0.
# If i=N-2, we check indices N-3, N-2, N-1. This corresponds to j=N-3.
# So, `j` runs from 0 to N-3 inclusive.
# The range `range(N-2)` generates indices from 0 up to N-3.
# If N < 3, N-2 < 1, range(N-2) is empty, correctly resulting in count 0.
for j in range(N - 2):
    # Check if the characters at indices j, j+1, j+2 match '#.#'
    if S[j] == '#' and S[j+1] == '.' and S[j+2] == '#':
        count += 1

# Print the result
print(count)