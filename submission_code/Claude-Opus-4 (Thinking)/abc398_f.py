# YOUR CODE HERE
S = input().strip()

for k in range(len(S)):
    if S[k:] == S[k:][::-1]:
        print(S + S[:k][::-1])
        break