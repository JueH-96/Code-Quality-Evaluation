n = int(input())
s = input().strip()

left_1s = [0] * n
for i in range(n):
    if s[i] == '1':
        left_1s[i] = left_1s[i-1] + 1 if i > 0 else 1
    else:
        left_1s[i] = 0

right_2s = [0] * n
for i in range(n-1, -1, -1):
    if s[i] == '2':
        right_2s[i] = right_2s[i+1] + 1 if i < n-1 else 1
    else:
        right_2s[i] = 0

max_len = 0
for i in range(n):
    if s[i] == '/':
        left = left_1s[i-1] if i > 0 else 0
        right = right_2s[i+1] if i < n-1 else 0
        current = 2 * min(left, right) + 1
        if current > max_len:
            max_len = current

print(max_len)