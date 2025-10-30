def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # Parse first line
    N = int(input_data[0])
    M = int(input_data[1])
    Sx = int(input_data[2])
    Sy = int(input_data[3])

    # Read house coordinates
    houses = input_data[4:4 + 2*N]
    idx = 0
    rows = {}
    cols = {}
    for _ in range(N):
        x = int(houses[idx])
        y = int(houses[idx+1])
        idx += 2
        if y not in rows:
            rows[y] = []
        rows[y].append(x)
        if x not in cols:
            cols[x] = []
        cols[x].append(y)

    # Sort row and column lists
    for y in rows:
        rows[y].sort()
    for x in cols:
        cols[x].sort()

    # Read instructions
    moves = input_data[4 + 2*N:]
    visited = set()  # to store visited house coordinates

    # Current position
    cur_x, cur_y = Sx, Sy
    ptr = 0
    for _ in range(M):
        d = moves[ptr]
        c = int(moves[ptr+1])
        ptr += 2

        if d == 'L':
            new_x = cur_x - c
            lo, hi = min(cur_x, new_x), max(cur_x, new_x)
            # Houses that match y = cur_y, x in [lo, hi]
            if cur_y in rows:
                arr = rows[cur_y]
                left_idx = bisect.bisect_left(arr, lo)
                right_idx = bisect.bisect_right(arr, hi)
                for xval in arr[left_idx:right_idx]:
                    if (xval, cur_y) not in visited:
                        visited.add((xval, cur_y))
            cur_x = new_x

        elif d == 'R':
            new_x = cur_x + c
            lo, hi = min(cur_x, new_x), max(cur_x, new_x)
            # Houses that match y = cur_y, x in [lo, hi]
            if cur_y in rows:
                arr = rows[cur_y]
                left_idx = bisect.bisect_left(arr, lo)
                right_idx = bisect.bisect_right(arr, hi)
                for xval in arr[left_idx:right_idx]:
                    if (xval, cur_y) not in visited:
                        visited.add((xval, cur_y))
            cur_x = new_x

        elif d == 'U':
            new_y = cur_y + c
            lo, hi = min(cur_y, new_y), max(cur_y, new_y)
            # Houses that match x = cur_x, y in [lo, hi]
            if cur_x in cols:
                arr = cols[cur_x]
                left_idx = bisect.bisect_left(arr, lo)
                right_idx = bisect.bisect_right(arr, hi)
                for yval in arr[left_idx:right_idx]:
                    if (cur_x, yval) not in visited:
                        visited.add((cur_x, yval))
            cur_y = new_y

        else:  # d == 'D'
            new_y = cur_y - c
            lo, hi = min(cur_y, new_y), max(cur_y, new_y)
            # Houses that match x = cur_x, y in [lo, hi]
            if cur_x in cols:
                arr = cols[cur_x]
                left_idx = bisect.bisect_left(arr, lo)
                right_idx = bisect.bisect_right(arr, hi)
                for yval in arr[left_idx:right_idx]:
                    if (cur_x, yval) not in visited:
                        visited.add((cur_x, yval))
            cur_y = new_y

    # Output final position and number of distinct houses visited
    print(cur_x, cur_y, len(visited))

# Do not forget to call main!
if __name__ == "__main__":
    main()