n = int(input())
s = input().strip()

if n % 2 == 0:
    print("No")
else:
    mid = (n - 1) // 2
    first = s[:mid]
    middle_char = s[mid]
    last = s[mid+1:]
    
    if all(c == '1' for c in first) and middle_char == '/' and all(c == '2' for c in last):
        print("Yes")
    else:
        print("No")