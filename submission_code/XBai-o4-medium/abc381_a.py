n = int(input())
s = input().strip()

if n % 2 == 0:
    print("No")
else:
    mid = (n - 1) // 2
    if s[mid] != '/':
        print("No")
    else:
        left_ok = all(c == '1' for c in s[:mid])
        right_ok = all(c == '2' for c in s[mid+1:])
        if left_ok and right_ok:
            print("Yes")
        else:
            print("No")