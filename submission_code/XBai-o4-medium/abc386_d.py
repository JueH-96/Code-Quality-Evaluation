import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    max_b = dict()
    min_w = dict()

    for _ in range(M):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        c = input[ptr]
        ptr += 1

        if c == 'B':
            if y in max_b:
                if x > max_b[y]:
                    max_b[y] = x
            else:
                max_b[y] = x
        else:  # 'W'
            if y in min_w:
                if x < min_w[y]:
                    min_w[y] = x
            else:
                min_w[y] = x

    # Collect all columns that have either B or W
    all_cols = set()
    for y in max_b:
        all_cols.add(y)
    for y in min_w:
        all_cols.add(y)

    for y in all_cols:
        lo = max_b.get(y, 0)
        up = min_w.get(y, N + 1)
        if lo > up - 1:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    main()