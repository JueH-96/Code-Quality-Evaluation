import sys
import bisect

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    S_x = int(data[ptr])
    ptr += 1
    S_y = int(data[ptr])
    ptr += 1

    houses_set = set()
    y_to_x = {}
    x_to_y = {}

    for _ in range(N):
        X = int(data[ptr])
        ptr += 1
        Y = int(data[ptr])
        ptr += 1
        houses_set.add((X, Y))
        if Y not in y_to_x:
            y_to_x[Y] = []
        y_to_x[Y].append(X)
        if X not in x_to_y:
            x_to_y[X] = []
        x_to_y[X].append(Y)

    for y in y_to_x:
        y_to_x[y].sort()
    for x in x_to_y:
        x_to_y[x].sort()

    directions = []
    for _ in range(M):
        D = data[ptr]
        ptr += 1
        C = int(data[ptr])
        ptr += 1
        directions.append((D, C))

    x, y = S_x, S_y
    encountered = set()

    for D, C in directions:
        if D == 'U':
            new_y = y + C
            new_x = x
            if y in y_to_x:
                x_list = y_to_x[y]
                left = bisect.bisect_left(x_list, min(x, new_x))
                right = bisect.bisect_right(x_list, max(x, new_x))
                for i in range(left, right):
                    encountered.add((x_list[i], y))
            y = new_y
        elif D == 'D':
            new_y = y - C
            new_x = x
            if y in y_to_x:
                x_list = y_to_x[y]
                left = bisect.bisect_left(x_list, min(x, new_x))
                right = bisect.bisect_right(x_list, max(x, new_x))
                for i in range(left, right):
                    encountered.add((x_list[i], y))
            y = new_y
        elif D == 'L':
            new_x = x - C
            new_y = y
            if x in x_to_y:
                y_list = x_to_y[x]
                bottom = bisect.bisect_left(y_list, min(y, new_y))
                top = bisect.bisect_right(y_list, max(y, new_y))
                for i in range(bottom, top):
                    encountered.add((x, y_list[i]))
            x = new_x
        elif D == 'R':
            new_x = x + C
            new_y = y
            if x in x_to_y:
                y_list = x_to_y[x]
                bottom = bisect.bisect_left(y_list, min(y, new_y))
                top = bisect.bisect_right(y_list, max(y, new_y))
                for i in range(bottom, top):
                    encountered.add((x, y_list[i]))
            x = new_x

    print(f"{x} {y} {len(encountered)}")

if __name__ == "__main__":
    main()