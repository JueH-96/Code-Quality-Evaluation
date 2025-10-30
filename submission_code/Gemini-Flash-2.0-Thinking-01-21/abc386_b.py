def solve():
    s = input()
    n = len(s)
    press_count = 0
    i = 0
    while i < n:
        if s[i] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            press_count += 1
            i += 1
        elif s[i] == '0':
            if i + 1 < n and s[i+1] == '0':
                press_count += 1
                i += 2
            else:
                press_count += 1
                i += 1
    print(press_count)

if __name__ == '__main__':
    solve()