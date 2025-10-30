import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    Sx = int(input[ptr]); ptr +=1
    Sy = int(input[ptr]); ptr +=1

    x_to_ys = defaultdict(list)
    y_to_xs = defaultdict(list)

    for _ in range(N):
        x = int(input[ptr]); ptr +=1
        y = int(input[ptr]); ptr +=1
        x_to_ys[x].append(y)
        y_to_xs[y].append(x)

    # Sort the lists for binary search
    for x in x_to_ys:
        x_to_ys[x].sort()
    for y in y_to_xs:
        y_to_xs[y].sort()

    current_x = Sx
    current_y = Sy

    houses_set = set()

    for _ in range(M):
        D = input[ptr]; ptr +=1
        C = int(input[ptr]); ptr +=1

        if D in ['U', 'D']:
            a = current_x
            if D == 'U':
                new_y = current_y + C
                y_min = current_y
                y_max = new_y
            else:  # 'D'
                new_y = current_y - C
                y_min = new_y
                y_max = current_y
            if a in x_to_ys:
                ys_list = x_to_ys[a]
                left = bisect_left(ys_list, y_min)
                right_idx = bisect_right(ys_list, y_max)
                for i in range(left, right_idx):
                    houses_set.add((a, ys_list[i]))
            current_y = new_y
        else:  # 'L' or 'R'
            b = current_y
            if D == 'R':
                new_x = current_x + C
                x_min = current_x
                x_max = new_x
            else:  # 'L'
                new_x = current_x - C
                x_min = new_x
                x_max = current_x
            if b in y_to_xs:
                xs_list = y_to_xs[b]
                left = bisect_left(xs_list, x_min)
                right_idx = bisect_right(xs_list, x_max)
                for i in range(left, right_idx):
                    houses_set.add((xs_list[i], b))
            current_x = new_x

    print(f"{current_x} {current_y} {len(houses_set)}")

if __name__ == '__main__':
    main()