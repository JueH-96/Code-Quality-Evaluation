from itertools import product

def check_overlap(a, b, x, ha, wa, hb, wb, hx, wx):
    for i, j in product(range(hx), range(wx)):
        if x[i][j] == '#' and a[i][j] == '.' and b[i][j] == '.':
            return False
        if x[i][j] == '.' and (a[i][j] == '#' or b[i][j] == '#'):
            return False
    return True

def main():
    ha, wa = map(int, input().split())
    a = [input() for _ in range(ha)]
    hb, wb = map(int, input().split())
    b = [input() for _ in range(hb)]
    hx, wx = map(int, input().split())
    x = [input() for _ in range(hx)]

    for i, j in product(range(hx), range(hx)):
        a_shifted = ['.' * j + row + '.' * (hx - wa - j) for row in a]
        b_shifted = ['.' * i + row + '.' * (hx - hb - i) for row in b]
        a_b_shifted = [a_row + b_row for a_row, b_row in zip(a_shifted, b_shifted)]
        if check_overlap(a_b_shifted, b_shifted, x, ha, wa, hb, wb, hx, wx):
            print('Yes')
            return
    print('No')

if __name__ == '__main__':
    main()