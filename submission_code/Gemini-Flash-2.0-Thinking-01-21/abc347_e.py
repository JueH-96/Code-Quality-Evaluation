def solve():
    n, q = map(int, input().split())
    x_values = list(map(int, input().split()))
    a = [0] * n
    s_set = set()
    for x in x_values:
        if x in s_set:
            s_set.remove(x)
        else:
            s_set.add(x)
        set_size = len(s_set)
        for element in s_set:
            a[element-1] += set_size
    print(*(a))

if __name__ == '__main__':
    solve()