# YOUR CODE HERE
S = input().strip()

# Find the character that differs from the first character
for i in range(len(S)):
    if S[i] != S[0]:
        print(i + 1)
        break