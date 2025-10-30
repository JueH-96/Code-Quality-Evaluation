n = int(input())
s = input().strip()

if n % 2 == 0:
    print("No")
else:
    mid = n // 2
    if s[mid] != '/':
        print("No")
    else:
        if all(c == '1' for c in s[:mid]) and all(c == '2' for c in s[mid+1:]):
            print("Yes")
        else:
            print("No")