def solve():
    n = int(input())
    s = input()
    if n < 3:
        print(0)
        return
    count = 0
    for i in range(n - 2):
        if s[i] == '#' and s[i+1] == '.' and s[i+2] == '#':
            count += 1
    print(count)

if __name__ == '__main__':
    solve()