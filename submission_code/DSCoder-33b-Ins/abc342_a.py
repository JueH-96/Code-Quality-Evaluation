# YOUR CODE HERE

S = input()

if S[0] == S[1]:
    for i in range(2, len(S)):
        if S[i] != S[0]:
            print(i+1)
            break
else:
    print(1)