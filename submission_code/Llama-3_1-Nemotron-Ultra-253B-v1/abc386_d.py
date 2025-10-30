def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    max_x_for_j = {}
    min_x_for_j = {}
    max_y_for_i = {}
    min_y_for_i = {}

    cells = []
    for _ in range(M):
        X = int(input[ptr])
        ptr += 1
        Y = int(input[ptr])
        ptr += 1
        C = input[ptr]
        ptr += 1
        cells.append((X, Y, C))

    for Y in range(1, N+1):
        max_x_for_j[Y] = 0
        min_x_for_j[Y] = N
    for X in range(1, N+1):
        max_y_for_i[X] = 0
        min_y_for_i[X] = N

    for X, Y, C in cells:
        if C == 'B':
            max_x_for_j[Y] = max(max_x_for_j[Y], X)
            max_y_for_i[X] = max(max_y_for_i[X], Y)
        else:
            min_x_for_j[Y] = min(min_x_for_j[Y], X-1)
            min_y_for_i[X] = min(min_y_for_i[X], Y-1)

    current_max_j = 0
    current_min_j = N
    j_low = {}
    j_high = {}
    for Y in reversed(range(1, N+1)):
        new_low = max(max_x_for_j[Y], current_max_j)
        new_high = min(min_x_for_j[Y], current_min_j)
        if new_low > new_high:
            print("No")
            return
        j_low[Y] = new_low
        j_high[Y] = new_high
        current_max_j = new_low
        current_min_j = new_high

    current_max_i = 0
    current_min_i = N
    i_low = {}
    i_high = {}
    for X in reversed(range(1, N+1)):
        new_low = max(max_y_for_i[X], current_max_i)
        new_high = min(min_y_for_i[X], current_min_i)
        if new_low > new_high:
            print("No")
            return
        i_low[X] = new_low
        i_high[X] = new_high
        current_max_i = new_low
        current_min_i = new_high

    for X, Y, C in cells:
        if C == 'B':
            if not (X <= j_high[Y] and Y <= i_high[X]):
                print("No")
                return
        else:
            if not (X > j_low[Y] or Y > i_low[X]):
                print("No")
                return

    max_k = 0
    for Y in range(1, N+1):
        max_k = max(max_k, j_low[Y])
    for X in range(1, N+1):
        max_k = max(max_k, i_low[X])

    min_k = N
    for Y in range(1, N+1):
        min_k = min(min_k, j_high[Y])
    for X in range(1, N+1):
        min_k = min(min_k, i_high[X])

    if max_k > min_k:
        print("No")
        return

    print("Yes")

main()