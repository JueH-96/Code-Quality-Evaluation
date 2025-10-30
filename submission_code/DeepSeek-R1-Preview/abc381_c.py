n = int(input())
s = input().strip()

left_ones = [0] * n
for i in range(n):
    if s[i] == '1':
        if i == 0:
            left_ones[i] = 1
        else:
            left_ones[i] = left_ones[i-1] + 1
    else:
        left_ones[i] = 0

right_twos = [0] * n
for i in range(n-1, -1, -1):
    if s[i] == '2':
        if i == n-1:
            right_twos[i] = 1
        else:
            right_twos[i] = right_twos[i+1] + 1
    else:
        right_twos[i] = 0

max_length = 0
positions = [i for i in range(n) if s[i] == '/']

for i in positions:
    left = left_ones[i-1] if i > 0 else 0
    right = right_twos[i+1] if i < n-1 else 0
    current = 2 * min(left, right) + 1
    if current > max_length:
        max_length = current

print(max_length)