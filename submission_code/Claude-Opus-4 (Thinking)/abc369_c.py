# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

count = 0

for l in range(N):
    r = l
    if l < N - 1:
        d = A[l + 1] - A[l]
        r = l + 1
        while r < N - 1 and A[r + 1] - A[r] == d:
            r += 1
    
    count += (r - l + 1)

print(count)