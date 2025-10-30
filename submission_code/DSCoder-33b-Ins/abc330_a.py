# YOUR CODE HERE

N, L = map(int, input().split())
A = list(map(int, input().split()))

passes = 0
for score in A:
    if score >= L:
        passes += 1

print(passes)