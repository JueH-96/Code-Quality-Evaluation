n = int(input())
s = input().strip()

left_ones = [0] * n
current = 0
for i in range(n):
    if s[i] == '1':
        current += 1
    else:
        current = 0
    left_ones[i] = current

right_twos = [0] * n
current = 0
for i in range(n-1, -1, -1):
    if s[i] == '2':
        current += 1
    else:
        current = 0
    right_twos[i] = current

max_len = 0

for i in range(n):
    if s[i] == '/':
        l = left_ones[i-1] if i > 0 else 0
        r = right_twos[i+1] if i < n-1 else 0
        sum_total = l + r
        if sum_total % 2 == 0:
            current = sum_total + 1
        else:
            option1 = (l - 1 + r) + 1 if l > 0 else 0
            option2 = (l + r - 1) + 1 if r > 0 else 0
            current = max(option1, option2)
        if current > max_len:
            max_len = current

print(max_len)