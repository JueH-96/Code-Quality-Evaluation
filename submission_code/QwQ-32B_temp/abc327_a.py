n = int(input())
s = input().strip()

found = False
for i in range(n - 1):
    pair = s[i] + s[i+1]
    if pair in ("ab", "ba"):
        print("Yes")
        found = True
        break

if not found:
    print("No")