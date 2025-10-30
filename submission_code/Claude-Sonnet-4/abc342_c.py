# YOUR CODE HERE
N = int(input())
S = input().strip()
Q = int(input())

for _ in range(Q):
    c, d = input().strip().split()
    S = S.replace(c, d)

print(S)