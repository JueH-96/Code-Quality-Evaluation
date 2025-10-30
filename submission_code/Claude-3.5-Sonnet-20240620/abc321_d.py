# YOUR CODE HERE
N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

total = 0
for a in A:
    for b in B:
        total += min(a + b, P)

print(total)