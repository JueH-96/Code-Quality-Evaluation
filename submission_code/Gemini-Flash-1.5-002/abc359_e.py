# YOUR CODE HERE
N = int(input())
H = list(map(int, input().split()))

A = [0] * (N + 1)
ans = []
count = 0
for i in range(1, N + 1):
    while A[i] == 0:
        count += 1
        A[0] += 1
        for j in range(1, N + 1):
            if A[j - 1] > A[j] and A[j - 1] > H[j - 1]:
                A[j - 1] -= 1
                A[j] += 1
    ans.append(count)

print(*ans)