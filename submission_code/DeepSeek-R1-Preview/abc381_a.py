n = int(input())
s = input().strip()

if n % 2 == 0:
    print("No")
else:
    mid = (n - 1) // 2
    if s[mid] != '/':
        print("No")
    else:
        valid = True
        # Check the part before '/'
        for i in range(mid):
            if s[i] != '1':
                valid = False
                break
        # Check the part after '/' only if the first part is valid
        if valid:
            for i in range(mid + 1, n):
                if s[i] != '2':
                    valid = False
                    break
        print("Yes" if valid else "No")