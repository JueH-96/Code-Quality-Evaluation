# YOUR CODE HERE
n = int(input())
keys = []
for i in range(n):
    a, s = input().split()
    keys.append((int(a), s))

left = keys[0][0]
right = keys[0][0]
ans = 0
left_pos = keys[0][0]
right_pos = keys[0][0]

for i in range(n):
    a, s = keys[i]
    if s == 'L':
        ans += abs(a - left_pos)
        left_pos = a
    else:
        ans += abs(a - right_pos)
        right_pos = a

print(ans)