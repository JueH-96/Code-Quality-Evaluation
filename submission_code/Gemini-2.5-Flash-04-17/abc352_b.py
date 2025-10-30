import sys

# Read S and T from standard input
# Use sys.stdin.readline().strip() for potentially faster input reading
# and to remove the trailing newline character.
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Initialize pointers for iterating through S and T
s_ptr = 0 # Points to the current character in S we are looking for
t_ptr = 0 # Points to the current position in T we are scanning

# List to store the 1-based indices of the correctly typed characters in T
correct_indices = []

# Iterate through each character in S
# For each character in S, we find its corresponding correctly typed character in T.
while s_ptr < len(S):
    # Search for the character S[s_ptr] in T.
    # Start the search from the current position t_ptr in T.
    # All characters in T from the previous match position up to
    # the current match position (exclusive) are considered mistakes.
    # The problem guarantees that the character S[s_ptr] will be found
    # at or after the current t_ptr position because T is generated
    # according to the defined procedure.
    while T[t_ptr] != S[s_ptr]:
        t_ptr += 1 # This character in T is a mistake, skip it by advancing t_ptr

    # T[t_ptr] is the correctly typed character in T that corresponds to S[s_ptr].
    # We found it at index t_ptr (0-based) in T.
    # The problem asks for the 1-based index.
    correct_indices.append(t_ptr + 1)

    # Move to the next character we need to find in S for the next iteration.
    s_ptr += 1

    # Move the T pointer forward. The search for the next S character (S[s_ptr])
    # will begin immediately after the character we just matched (T[t_ptr]).
    t_ptr += 1 # This becomes the starting index for the search in the next outer loop iteration

# Print the collected 1-based indices, separated by spaces.
# The indices are naturally collected in ascending order because t_ptr only increases.
print(*correct_indices)