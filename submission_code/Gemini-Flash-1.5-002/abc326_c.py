# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
a.sort()
max_gifts = 0
left = 0
right = 0
while right < n:
    while right < n and a[right] < a[left] + m:
        right += 1
    max_gifts = max(max_gifts, right - left)
    left += 1
print(max_gifts)