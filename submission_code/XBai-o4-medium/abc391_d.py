import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1

    blocks = []
    columns = [[] for _ in range(W+1)]  # 1-based

    for _ in range(N):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        blocks.append((x, y))
        columns[x].append(y)

    has_initial_bottom = [False] * (W + 1)
    initial_count = [0] * (W + 1)

    for x in range(1, W+1):
        initial_count[x] = len(columns[x])
        for y in columns[x]:
            if y == 1:
                has_initial_bottom[x] = True

    all_has = True
    for x in range(1, W+1):
        if not has_initial_bottom[x]:
            all_has = False
            break

    initial_min = 0
    if all_has:
        initial_min = min(initial_count[x] for x in range(1, W+1))
    else:
        initial_min = 0

    # Precompute r_i for each block
    r = [0] * N
    for i in range(N):
        x, y = blocks[i]
        sorted_y = sorted(columns[x])
        pos = bisect.bisect_right(sorted_y, y)
        r[i] = pos

    Q = int(input[ptr])
    ptr += 1
    output = []
    for _ in range(Q):
        T = int(input[ptr])
        ptr += 1
        A = int(input[ptr]) - 1  # 0-based
        ptr += 1

        if all_has:
            k = min(T, initial_min)
            if k >= r[A]:
                output.append("No")
            else:
                output.append("Yes")
        else:
            output.append("Yes")

    print('
'.join(output))

if __name__ == '__main__':
    main()