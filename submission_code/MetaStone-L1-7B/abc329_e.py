n, m = map(int, input().split())
s = input().strip()
t = input().strip()

for j in range(n):
    found = False
    start = max(0, j - m + 1)
    end = min(j, n - m)
    for i in range(start, end + 1):
        if t[j - i] == s[j]:
            found = True
            break
    if not found:
        print("No")
        exit()
print("Yes")