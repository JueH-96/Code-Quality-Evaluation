n = int(input())
s = input().strip()

left_ones = [0] * n
for i in range(n):
    if s[i] == '1':
        left_ones[i] = left_ones[i-1] + 1 if i > 0 else 1
    else:
        left_ones[i] = 0

right_twos = [0] * n
for i in range(n-1, -1, -1):
    if s[i] == '2':
        right_twos[i] = right_twos[i+1] + 1 if i < n-1 else 1
    else:
        right_twos[i] = 0

max_len = 0
for i in range(n):
    if s[i] == '/':
        left_count = left_ones[i-1] if i > 0 else 0
        right_count = right_twos[i+1] if i < n-1 else 0
        current = 2 * min(left_count, right_count) + 1
        if current > max_len:
            max_len = current

print(max_len)