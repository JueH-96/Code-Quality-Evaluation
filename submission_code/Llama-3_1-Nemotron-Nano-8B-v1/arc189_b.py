n, *rest = map(int, open(0).read().split())
x = rest[:n]
total = sum(x)
for i in range(n - 3):
    a, b, c, d = x[i], x[i+1], x[i+2], x[i+3]
    if a + d < b + c:
        total += 2 * (a + d - b - c)
print(total)