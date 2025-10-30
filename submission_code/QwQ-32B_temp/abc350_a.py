s = input().strip()
number_part = s[3:]
num = int(number_part)
if 1 <= num <= 349 and num != 316:
    print("Yes")
else:
    print("No")