# YOUR CODE HERE
S = input().strip()
rev_S = S[::-1]

for i in range(len(S)):
    if S.endswith(rev_S[:i+1]):
        result = S + rev_S[i+1:]
        print(result)
        break