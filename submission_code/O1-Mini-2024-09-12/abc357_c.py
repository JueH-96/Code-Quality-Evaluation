import sys

def solve():
    N = int(sys.stdin.read())
    size = 3**N
    for i in range(size):
        line = []
        ti, tj = i, 0
        for j in range(size):
            x, y = i, j
            is_white = False
            while x > 0 or y > 0:
                if x % 3 == 1 and y % 3 == 1:
                    is_white = True
                    break
                x //= 3
                y //= 3
            if is_white:
                line.append('.')
            else:
                line.append('#')
        print(''.join(line))