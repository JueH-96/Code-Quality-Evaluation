n = int(input())
s = input().strip()

if n % 2 == 0:
    print("No")
else:
    middle = (n - 1) // 2
    if s[middle] != '/':
        print("No")
    else:
        valid = True
        # Check all characters before middle are '1's
        for i in range(middle):
            if s[i] != '1':
                valid = False
                break
        # Check all characters after middle are '2's
        if valid:
            for i in range(middle + 1, n):
                if s[i] != '2':
                    valid = False
                    break
        print("Yes" if valid else "No")