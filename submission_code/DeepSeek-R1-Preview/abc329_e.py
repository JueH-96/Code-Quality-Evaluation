n, m = map(int, input().split())
s = input().strip()
t = input().strip()

for i in range(1, n + 1):
    # Calculate the range of possible j for current i
    a = max(1, i - (m - 1))
    b = min(i, n - m + 1)
    found = False
    # Iterate from rightmost to leftmost j
    for j in range(b, a - 1, -1):
        t_index = i - j
        if t_index < 0 or t_index >= m:
            continue
        if t[t_index] == s[i - 1]:
            found = True
            break
    if not found:
        print("No")
        exit()
print("Yes")