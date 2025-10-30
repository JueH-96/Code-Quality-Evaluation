s = input()
num_str = s[3:]
num = int(num_str)

if (1 <= num <= 314) or (num == 315) or (317 <= num <= 348) or (num == 349):
    print("Yes")
else:
    print("No")