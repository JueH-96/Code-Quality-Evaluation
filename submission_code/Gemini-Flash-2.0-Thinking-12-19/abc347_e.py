def solve():
    n, q = map(int, input().split())
    x_values = list(map(int, input().split()))
    a = [0] * n
    s = set()
    for x in x_values:
        if x in s:
            s.remove(x)
        else:
            s.add(x)
        s_size = len(s)
        for j in s:
            if 1 <= j <= n:
                a[j-1] += s_size
                
    print(*(a))

if __name__ == '__main__':
    solve()