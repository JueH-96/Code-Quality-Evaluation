import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    Sx = int(input[ptr])
    ptr +=1
    Sy = int(input[ptr])
    ptr +=1

    x_to_ys = {}
    y_to_xs = {}

    for _ in range(N):
        X = int(input[ptr])
        ptr +=1
        Y = int(input[ptr])
        ptr +=1
        if X not in x_to_ys:
            x_to_ys[X] = []
        bisect.insort(x_to_ys[X], Y)
        if Y not in y_to_xs:
            y_to_xs[Y] = []
        bisect.insort(y_to_xs[Y], X)

    visited = set()
    final_x = Sx
    final_y = Sy
    current_x, current_y = Sx, Sy

    for _ in range(M):
        C = input[ptr]
        ptr +=1
        R = int(input[ptr])
        ptr +=1
        C_dir = C
        if C_dir == 'U' or C_dir == 'D':
            X_start = current_x
            Y_start = current_y
            X_end = X_start
            Y_end = Y_start + R if C_dir == 'U' else Y_start - R
            Y_min = min(Y_start, Y_end)
            Y_max = max(Y_start, Y_end)
            X = X_start
            Y_list = x_to_ys.get(X, [])
            left = bisect.bisect_left(Y_list, Y_min)
            right_idx = bisect.bisect_right(Y_list, Y_max)
            for i in range(left, right_idx):
                Y = Y_list[i]
                if (X, Y) not in visited:
                    visited.add((X, Y))
            current_y = Y_end
        else:
            Y_start = current_y
            X_start = current_x
            Y_end = Y_start
            X_end = X_start + R if C_dir == 'R' else X_start - R
            X_min = min(X_start, X_end)
            X_max = max(X_start, X_end)
            Y = Y_start
            X_list = y_to_xs.get(Y, [])
            left = bisect.bisect_left(X_list, X_min)
            right_idx = bisect.bisect_right(X_list, X_max)
            for i in range(left, right_idx):
                X = X_list[i]
                if (X, Y) not in visited:
                    visited.add((X, Y))
            current_x = X_end

    print(len(visited), current_x, current_y)

if __name__ == '__main__':
    main()