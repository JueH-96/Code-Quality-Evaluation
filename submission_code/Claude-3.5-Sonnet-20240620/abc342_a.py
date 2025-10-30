# YOUR CODE HERE
S = input().strip()
for i, char in enumerate(S, 1):
    if S.count(char) == 1:
        print(i)
        break