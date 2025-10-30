# YOUR CODE HERE
a, m, l, r = map(int, input().split())

count = 0

start = (l - a) // m
end = (r - a) // m

for k in range(start, end + 1):
    tree_pos = a + k * m
    if l <= tree_pos <= r:
        count += 1

print(count)