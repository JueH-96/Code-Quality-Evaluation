n = int(input())
s = input().strip()

left_ones = [0] * n
if n > 0:
    left_ones[0] = 1 if s[0] == '1' else 0
for i in range(1, n):
    if s[i] == '1':
        left_ones[i] = left_ones[i-1] + 1
    else:
        left_ones[i] = 0

right_twos = [0] * n
if n > 0:
    right_twos[-1] = 1 if s[-1] == '2' else 0
for i in range(n-2, -1, -1):
    if s[i] == '2':
        right_twos[i] = right_twos[i+1] + 1
    else:
        right_twos[i] = 0

max_len = 0
for i in range(n):
    if s[i] == '/':
        a = left_ones[i-1] if i > 0 else 0
        b = right_twos[i+1] if i < n-1 else 0
        current = 2 * min(a, b) + 1
        if current > max_len:
            max_len = current

print(max_len)