n, m = map(int, input().split())
x = list(map(int, input().split()))

sum_total = 0
for i in range(m):
    a = x[i]
    b = x[(i + 1) % m]
    d = min((b - a) % n, (a - b) % n)
    sum_total += d

delta_left = [0] * n
delta_right = [0] * n

for i in range(m):
    a = x[i]
    b = x[(i + 1) % m]
    
    if a <= b - 1:
        delta_left[a - 1] += 1
        if b <= n:
            delta_left[b - 1] -= 1
        else:
            pass
    
    if a > b:
        delta_right[b - 1] += 1
        delta_right[a - 1] -= 1

left = [0] * n
current = 0
for i in range(n):
    current += delta_left[i]
    left[i] = current

right = [0] * n
current = 0
for i in range(n):
    current += delta_right[i]
    right[i] = current

min_sum = float('inf')
for i in range(n):
    cnt = left[i] + right[i]
    total = sum_total + cnt
    if total < min_sum:
        min_sum = total

print(min_sum)