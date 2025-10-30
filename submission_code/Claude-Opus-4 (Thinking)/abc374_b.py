# YOUR CODE HERE
S = input()
T = input()

if S == T:
    print(0)
else:
    # Find the first position where they differ
    i = 0
    while i < len(S) and i < len(T) and S[i] == T[i]:
        i += 1
    print(i + 1)  # Convert to 1-indexed