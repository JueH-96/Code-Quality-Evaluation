s = input().strip()
num_part = s[3:]
n = int(num_part)
if 1 <= n <= 349 and n != 316:
    print("Yes")
else:
    print("No")