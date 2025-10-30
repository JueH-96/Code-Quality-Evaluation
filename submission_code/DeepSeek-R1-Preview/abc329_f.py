import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    C = list(map(int, input[ptr:ptr + N]))
    ptr += N

    # Initialize box_sets
    box_sets = {}
    for i in range(1, N + 1):
        box_sets[i] = {C[i - 1]}

    for _ in range(Q):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1

        if not box_sets[a]:
            print(len(box_sets[b]))
            continue

        # Move all colors from a to b
        added = 0
        a_colors = box_sets[a]
        b_colors = box_sets[b]
        for color in a_colors:
            if color not in b_colors:
                b_colors.add(color)
        print(len(b_colors))
        a_colors.clear()

if __name__ == "__main__":
    main()