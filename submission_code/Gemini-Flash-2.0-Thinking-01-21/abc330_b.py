def solve():
    n, l, r = map(int, input().split())
    a = list(map(int, input().split()))
    x_sequence = []
    for val_a in a:
        if val_a < l:
            x_sequence.append(l)
        elif val_a > r:
            x_sequence.append(r)
        else:
            x_sequence.append(val_a)
    print(*(x_sequence))

if __name__ == '__main__':
    solve()