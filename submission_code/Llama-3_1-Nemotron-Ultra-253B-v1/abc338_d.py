n, m = map(int, input().split())
x = list(map(int, input().split()))

sum_d = 0
diff = [0] * (n + 2)  # 1-based indexing, diff up to n+1

prev = x[0]
for i in range(1, m):
    a = prev
    b = x[i]
    d_clockwise = (b - a) % n
    d_counter = (a - b) % n
    if d_clockwise < d_counter:
        d = d_clockwise
        s, e = a, b
    else:
        d = d_counter
        s, e = b, a
    sum_d += d
    value = n - 2 * d
    if s <= e:
        diff[s] += value
        diff[e] -= value
    else:
        diff[s] += value
        diff[n + 1] -= value
        diff[1] += value
        diff[e] -= value
    prev = b

# Compute delta for each bridge
delta = [0] * (n + 1)
current = 0
for i in range(1, n + 1):
    current += diff[i]
    delta[i] = current

min_total = sum_d + min(delta[1:n+1])
print(min_total)