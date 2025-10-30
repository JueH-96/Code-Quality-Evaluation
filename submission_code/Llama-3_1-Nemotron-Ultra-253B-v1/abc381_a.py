n = int(input())
s = input().strip()

if n % 2 == 0:
    print("No")
else:
    mid = n // 2
    first = s[:mid]
    second = s[mid+1:]
    if (all(c == '1' for c in first) and
        s[mid] == '/' and
        all(c == '2' for c in second)):
        print("Yes")
    else:
        print("No")