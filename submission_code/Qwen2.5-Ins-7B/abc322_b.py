# YOUR CODE HERE
N, M = map(int, input().split())
S = input().strip()
T = input().strip()

if T.startswith(S) and T.endswith(S):
    print(0)
elif T.startswith(S) and not T.endswith(S):
    print(1)
elif not T.startswith(S) and T.endswith(S):
    print(2)
else:
    print(3)