# YOUR CODE HERE

S = input()

for i in range(len(S)):
    if S[i] != S[0]:
        print(i+1)
        break