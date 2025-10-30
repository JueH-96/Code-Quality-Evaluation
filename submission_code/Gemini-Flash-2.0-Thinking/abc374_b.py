s = input()
t = input()

if s == t:
    print(0)
else:
    n = len(s)
    m = len(t)
    min_len = min(n, m)
    found_difference = False
    for i in range(min_len):
        if s[i] != t[i]:
            print(i + 1)
            found_difference = True
            break
    if not found_difference:
        if n < m:
            print(n + 1)
        elif m < n:
            print(m + 1)