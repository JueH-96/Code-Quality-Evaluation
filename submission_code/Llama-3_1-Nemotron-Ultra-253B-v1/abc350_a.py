s = input().strip()
num = int(s[3:])
if (1 <= num <= 315) or (317 <= num <= 349):
    print("Yes")
else:
    print("No")