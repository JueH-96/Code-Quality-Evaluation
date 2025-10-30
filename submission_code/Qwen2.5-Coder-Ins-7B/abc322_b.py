# YOUR CODE HERE
N, M = map(int, input().split())
S = input()
T = input()

if T.startswith(S) and T.endswith(S):
    print(0)
elif T.startswith(S):
    print(1)
elif T.endswith(S):
    print(2)
else:
    print(3)