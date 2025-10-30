def is_11_22_string(s):
    n = len(s)
    if n % 2 == 0:
        return False
    mid = (n + 1) // 2
    if s[mid - 1] != '/':
        return False
    for i in range(mid - 1):
        if s[i] != '1':
            return False
    for i in range(mid, n):
        if s[i] != '2':
            return False
    return True

def solve():
    n = int(input())
    s = input()
    max_len = 0
    for i in range(n):
        for j in range(i, n):
            sub = s[i:j+1]
            if is_11_22_string(sub):
                max_len = max(max_len, len(sub))
    print(max_len)

solve()