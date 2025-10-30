# YOUR CODE HERE
S = input().strip()

# Find the first character that is different from the first character
for i in range(1, len(S)):
    if S[i] != S[0]:
        print(i + 1)
        break