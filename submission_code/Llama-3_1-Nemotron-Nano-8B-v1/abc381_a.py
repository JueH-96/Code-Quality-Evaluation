n = int(input())
s = input().strip()

if n % 2 == 0:
    print("No")
else:
    m = (n + 1) // 2
    first_valid = all(c == '1' for c in s[:m-1])
    middle_valid = s[m-1] == '/'
    third_valid = all(c == '2' for c in s[m:])
    if first_valid and middle_valid and third_valid:
        print("Yes")
    else:
        print("No")