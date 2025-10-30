# YOUR CODE HERE
S = input().strip()
for i in range(len(S)):
    if S[i] != S[0] or S[i] != S[-1]:
        print(i+1)
        break