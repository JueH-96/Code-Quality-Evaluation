import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N, W = int(input[ptr]), int(input[ptr+1])
    ptr += 2

    columns = [[] for _ in range(W+1)]  # 1-based indexing
    blocks = []
    for i in range(N):
        X = int(input[ptr])
        Y = int(input[ptr+1])
        ptr += 2
        columns[X].append(Y)
        blocks.append((X, Y))

    # Sort each column's list of Y's
    for x in range(1, W+1):
        columns[x].sort()

    removal_times = []
    current_time = 0
    R = 0

    while True:
        max_delta = -1
        possible = True
        for x in range(1, W+1):
            ys = columns[x]
            target = R + 1
            idx = bisect.bisect_left(ys, target)
            if idx >= len(ys):
                possible = False
                break
            Y_val = ys[idx]
            delta_x = Y_val - R - 1
            if delta_x > max_delta:
                max_delta = delta_x
        if not possible:
            break
        next_removal_time = current_time + max_delta + 1
        removal_times.append(next_removal_time)
        current_time = next_removal_time
        R += 1

    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        T = int(input[ptr])
        A = int(input[ptr+1])
        ptr += 2
        X, Y = blocks[A-1]
        R_needed = Y - 1
        if R_needed > len(removal_times):
            print("Yes")
        else:
            if R_needed == 0:
                if len(removal_times) == 0:
                    print("Yes")
                else:
                    removal_time = removal_times[0]
                    if T < removal_time:
                        print("Yes")
                    else:
                        print("No")
            else:
                if R_needed - 1 < len(removal_times):
                    removal_time = removal_times[R_needed - 1]
                    if T < removal_time:
                        print("Yes")
                    else:
                        print("No")
                else:
                    print("Yes")

if __name__ == "__main__":
    main()