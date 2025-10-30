# YOUR CODE HERE

N = int(input())
P = list(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    A, B = map(int, input().split())
    print(P[A-1])