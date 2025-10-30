s = input().strip()
num = int(s[3:])

if num <= 315:
    print("Yes")
elif 317 <= num <= 349:
    print("Yes")
else:
    print("No")