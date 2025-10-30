n = int(input())
q = list(map(int, input().split()))
a = list(map(int, input().split()))
b = list(map(int, input().split()))

x_max = float('inf')
for i in range(n):
    ai = a[i]
    if ai > 0:
        current = q[i] // ai
        if current < x_max:
            x_max = current

max_total = 0
for x in range(x_max + 1):
    rem = [q[i] - a[i] * x for i in range(n)]
    possible_ys = []
    for i in range(n):
        bi = b[i]
        if bi > 0:
            possible_ys.append(rem[i] // bi)
    y_candidate = min(possible_ys)
    total = x + y_candidate
    if total > max_total:
        max_total = total

print(max_total)