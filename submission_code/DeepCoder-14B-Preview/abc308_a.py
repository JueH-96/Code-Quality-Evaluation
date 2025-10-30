s = list(map(int, input().split()))
if len(s) != 8:
    print("No")
    exit()

if not all(x % 25 == 0 for x in s):
    print("No")
    exit()

if not all(100 <= x <= 675 for x in s):
    print("No")
    exit()

for i in range(7):
    if s[i] > s[i + 1]:
        print("No")
        exit()

print("Yes")