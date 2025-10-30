N, M = map(int, input().split())
A = list(map(int, input().split()))

A.sort()
max_gifts = 0
left = 0

for right in range(N):
    # Slide left pointer while window size > M
    while A[right] - A[left] >= M:
        left += 1
    max_gifts = max(max_gifts, right - left + 1)

print(max_gifts)