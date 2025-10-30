def solve():
    n = int(input())
    s = input()
    ans = 0
    for i in range(n):
        for j in range(i, n):
            sub = s[i:j+1]
            if is_11_22_string(sub):
                ans = max(ans, len(sub))
    print(ans)

def is_11_22_string(t):
    if len(t) % 2 == 0:
        return False
    mid = (len(t) + 1) // 2
    if t[mid - 1] != '/':
        return False
    for i in range(mid - 1):
        if t[i] != '1':
            return False
    for i in range(mid, len(t)):
        if t[i] != '2':
            return False
    return True

solve()