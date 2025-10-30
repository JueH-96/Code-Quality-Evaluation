n = int(input())
s = input().strip()

left_ones = [0] * n
if n >= 1:
    left_ones[0] = 1 if s[0] == '1' else 0
for i in range(1, n):
    if s[i] == '1':
        left_ones[i] = left_ones[i-1] + 1
    else:
        left_ones[i] = 0

right_twos = [0] * n
if n >= 1:
    right_twos[-1] = 1 if s[-1] == '2' else 0
for i in range(n-2, -1, -1):
    if s[i] == '2':
        right_twos[i] = right_twos[i+1] + 1
    else:
        right_twos[i] = 0

max_len = 1  # At least one '/' exists
for i in range(n):
    if s[i] == '/':
        left = left_ones[i-1] if i > 0 else 0
        right = right_twos[i+1] if i < n-1 else 0
        current_len = 2 * min(left, right) + 1
        if current_len > max_len:
            max_len = current_len

print(max_len)