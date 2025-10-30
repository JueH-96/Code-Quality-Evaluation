def calculate_mex(x, y, z):
    seen = {x, y, z}
    if 0 not in seen:
        return 0
    if 1 not in seen:
        return 1
    if 2 not in seen:
        return 2
    return 3

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = input()
    mex_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if s[i] == 'M' and s[j] == 'E' and s[k] == 'X':
                    mex_val = calculate_mex(a[i], a[j], a[k])
                    mex_sum += mex_val
    print(mex_sum)

if __name__ == '__main__':
    solve()