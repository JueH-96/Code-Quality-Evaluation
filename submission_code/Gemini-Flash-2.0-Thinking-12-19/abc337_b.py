def solve():
    s = input()
    n = len(s)
    i1 = 0
    while i1 < n and s[i1] == 'A':
        i1 += 1
    i2 = i1
    while i2 < n and s[i2] == 'B':
        i2 += 1
    i3 = i2
    while i3 < n and s[i3] == 'C':
        i3 += 1
    if i3 == n:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()