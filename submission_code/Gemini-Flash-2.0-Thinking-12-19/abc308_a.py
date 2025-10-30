s = list(map(int, input().split()))

valid = True

for i in range(len(s)):
    if not (100 <= s[i] <= 675):
        valid = False
        break
    if s[i] % 25 != 0:
        valid = False
        break
    if i > 0:
        if s[i] < s[i-1]:
            valid = False
            break

if valid:
    print("Yes")
else:
    print("No")