# Read the integer N
N = int(input())
# Read the string S
S = input()

# Initialize a counter for the number of satisfying integers i
ans = 0

# Iterate with k as the 0-indexed start of a 3-character window in S.
# S[k] corresponds to seat k+1 (1-indexed).
# The problem is interested in i (1-indexed) from 1 to N-2.
# If k is the 0-indexed start of the pattern in S, then i = k+1.
# So, k ranges from (1-1)=0 to (N-2)-1 = N-3.
# The Python loop `for k in range(N - 2)` iterates k from 0 to N-3.
# If N < 3, then N-2 will be less than 1.
# e.g. N=1 -> range(-1) is empty. N=2 -> range(0) is empty.
# This correctly results in ans=0 for N < 3, as no 3-seat pattern can form.
for k in range(N - 2):
    # Check if the characters S[k], S[k+1], S[k+2] match the pattern '#.#'.
    # S[k] corresponds to seat i (occupied)
    # S[k+1] corresponds to seat i+1 (unoccupied)
    # S[k+2] corresponds to seat i+2 (occupied)
    if S[k] == '#' and S[k+1] == '.' and S[k+2] == '#':
        ans += 1

# Print the final count
print(ans)