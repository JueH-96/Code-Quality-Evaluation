n = input().strip()
if len(n) == 1:
    print("Yes")
else:
    digits = [int(c) for c in n]
    for i in range(len(digits) - 1):
        if digits[i] <= digits[i + 1]:
            print("No")
            break
    else:
        print("Yes")