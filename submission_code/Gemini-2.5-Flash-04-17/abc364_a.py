# YOUR CODE HERE
import sys

# Read N, the number of dishes
N = int(sys.stdin.readline())

# Read the N dish types into a list
# Each element is a string 'sweet' or 'salty'
S = [sys.stdin.readline().strip() for _ in range(N)]

# Takahashi eats dishes sequentially S[0], S[1], ..., S[N-1].
# If he eats S[i] and then S[i+1], and both are 'sweet', he feels sick after eating S[i+1].
# After feeling sick, he cannot eat any more dishes.
# He can eat all N dishes only if he does not stop eating before finishing the N-th dish (S[N-1]).
# He stops after eating S[i+1] if S[i] and S[i+1] are sweet.
# He stops prematurely if this happens AND S[i+1] is NOT the index of the last dish (N-1).
# This means he stops prematurely if S[i] == 'sweet' and S[i+1] == 'sweet' for any index i
# such that i+1 < N-1.
# The condition i+1 < N-1 is equivalent to i < N-2.
# So, failure occurs if S[i] == 'sweet' and S[i+1] == 'sweet' for any index i
# in the range from 0 up to N-3 inclusive.
# The Python range for this is range(N-2).

failed = False
# Loop through possible indices `i` that could start a sweet-sweet pair (S[i], S[i+1])
# that would cause premature failure.
# These are indices i from 0 up to N-3.
# The loop `for i in range(N - 2)` covers these indices.
# If N < 2, N-2 < 0, range(N-2) is empty, loop doesn't run, `failed` remains False, correctly yields "Yes".
for i in range(N - 2):
    # Check the pair (S[i], S[i+1])
    if S[i] == 'sweet' and S[i+1] == 'sweet':
        # Found a sweet-sweet pair at indices i and i+1, where i is in range(N-2).
        # This implies i <= N-3, so i+1 <= N-2.
        # Since i+1 <= N-2 and N-2 < N-1, we have i+1 < N-1.
        # A sweet-sweet pair at indices i, i+1 where i+1 < N-1 causes premature stopping.
        failed = True
        break # Found a failure condition, no need to check further

# If the loop finished without setting `failed` to True, it means either:
# - N < 2 (the loop range(N-2) is empty)
# - No sweet-sweet pair exists at indices (i, i+1) where i < N-2.
#   This includes cases where sweet-sweet pairs only exist at the very end (i.e., i=N-2),
#   or where no sweet-sweet pairs exist at all.
# In all these cases where `failed` is False, he successfully eats all N dishes.

if failed:
    print("No")
else:
    print("Yes")