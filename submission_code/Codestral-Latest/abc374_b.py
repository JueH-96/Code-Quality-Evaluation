# YOUR CODE HERE
S = input().strip()
T = input().strip()

min_length = min(len(S), len(T))

for i in range(min_length):
    if S[i] != T[i]:
        print(i + 1)
        break
else:
    if len(S) == len(T):
        print(0)
    else:
        print(min_length + 1)