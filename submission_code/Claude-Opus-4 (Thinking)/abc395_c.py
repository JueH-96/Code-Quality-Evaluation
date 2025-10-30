# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

last_pos = {}
min_length = float('inf')

for i in range(n):
    if a[i] in last_pos:
        length = i - last_pos[a[i]] + 1
        min_length = min(min_length, length)
    last_pos[a[i]] = i

if min_length == float('inf'):
    print(-1)
else:
    print(min_length)