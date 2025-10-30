n = int(input())
x = list(map(int, input().split()))

for i in range(n - 4, -1, -1):
    a = x[i] + x[i + 3]
    b = x[i + 1] + x[i + 2]
    if b > a:
        m = (x[i] + x[i + 3]) / 2
        new_a = 2 * m - x[i + 1]
        new_b = 2 * m - x[i + 2]
        if new_a > new_b:
            new_a, new_b = new_b, new_a
        x[i + 1] = new_a
        x[i + 2] = new_b

print(sum(x))