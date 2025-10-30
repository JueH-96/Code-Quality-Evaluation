# YOUR CODE HERE
S = input().strip()
# Determine the majority character
if S.count(S[0]) >= 2:
    majority_char = S[0]
else:
    majority_char = S[1]
# Find the index of the differing character
for i, char in enumerate(S, 1):
    if char != majority_char:
        print(i)
        break