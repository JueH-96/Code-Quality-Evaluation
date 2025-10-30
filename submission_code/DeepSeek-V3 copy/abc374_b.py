# YOUR CODE HERE
S = input().strip()
T = input().strip()

min_length = min(len(S), len(T))
for i in range(min_length):
    if S[i] != T[i]:
        print(i + 1)
        exit()
if len(S) != len(T):
    print(min_length + 1)
else:
    print(0)