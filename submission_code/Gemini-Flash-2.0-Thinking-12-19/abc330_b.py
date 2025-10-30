def solve():
    n, l, r = map(int, input().split())
    a = list(map(int, input().split()))
    x_values = []
    for val_a in a:
        x_i = min(max(l, val_a), r)
        x_values.append(x_i)
    print(*(x_values))

if __name__ == '__main__':
    solve()