s = input().strip()
n = len(s)

if n % 2 != 0:
    print("No")
else:
    valid = True
    for i in range(0, n, 2):
        if s[i] != s[i+1]:
            valid = False
            break
    if valid:
        from collections import Counter
        cnt = Counter(s)
        for v in cnt.values():
            if v != 2:
                valid = False
                break
    print("Yes" if valid else "No")