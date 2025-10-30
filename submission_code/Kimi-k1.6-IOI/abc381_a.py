n = int(input())
s = input().strip()

if n % 2 == 0:
    print("No")
else:
    m = (n - 1) // 2
    if s[m] != '/':
        print("No")
    else:
        valid = True
        # Check all characters before '/' are '1's
        for i in range(m):
            if s[i] != '1':
                valid = False
                break
        if valid:
            # Check all characters after '/' are '2's
            for i in range(m + 1, n):
                if s[i] != '2':
                    valid = False
                    break
        print("Yes" if valid else "No")