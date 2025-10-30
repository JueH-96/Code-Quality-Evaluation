# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

available = list(range(1, n + 1))
for x in a:
    available.remove(x)

num_pairs = (2 * n - k) // 2
weirdness = 0
left = 0
right = len(available) - 1
if len(available) > 0:
    while left < right and num_pairs > 0:
        weirdness += available[right] - available[left]
        left += 1
        right -= 1
        num_pairs -= 1

print(weirdness)