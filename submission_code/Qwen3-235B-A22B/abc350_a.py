S = input().strip()
number_part = S[3:]
n = int(number_part)
if 1 <= n <= 349 and n != 316:
    print("Yes")
else:
    print("No")