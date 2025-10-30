# YOUR CODE HERE
S = input()
num_upper = sum(c.isupper() for c in S)
if 2 * num_upper > len(S):
    print(S.upper())
else:
    print(S.lower())