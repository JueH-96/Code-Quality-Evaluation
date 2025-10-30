n, s = map(int, input().split())
a = list(map(int, input().split()))

found = False
for i in range(n):
    current_sum = 0
    for j in range(i, 2 * n):
        current_sum += a[j % n]
        if current_sum == s:
            found = True
            break
    if found:
        break

if found:
    print("Yes")
else:
    print("No")