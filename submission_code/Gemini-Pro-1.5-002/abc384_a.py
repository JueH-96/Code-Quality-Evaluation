n, c1, c2 = input().split()
n = int(n)
s = input()
ans = ""
for char in s:
    if char == c1:
        ans += c1
    else:
        ans += c2
print(ans)