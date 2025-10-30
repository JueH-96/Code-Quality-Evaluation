def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1

    C = list(map(int, data[ptr:ptr+N]))
    ptr += N

    current_box = {}
    for i in range(N):
        c = C[i]
        current_box[c] = i + 1  # Boxes are 1-based

    box_counts = [0] * (N + 1)  # 1-based
    for i in range(N):
        c = C[i]
        box_counts[current_box[c]] += 1

    for _ in range(Q):
        a = int(data[ptr])
        ptr += 1
        b = int(data[ptr])
        ptr += 1

        if a == b:
            print(box_counts[b])
            continue

        # Collect all colors in box a
        a_set = set()
        for c in current_box:
            if current_box[c] == a:
                a_set.add(c)

        new_colors = 0
        for c in a_set:
            if current_box[c] != b:
                current_box[c] = b
                new_colors += 1

        box_counts[b] += new_colors

        print(box_counts[b])

if __name__ == '__main__':
    main()