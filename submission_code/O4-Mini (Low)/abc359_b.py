def main():
    import sys
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    arr = list(map(int, data[1:]))

    first_pos = {}
    count = 0

    for idx, color in enumerate(arr):
        if color not in first_pos:
            first_pos[color] = idx
        else:
            # Check if exactly one person is between the two occurrences
            if idx - first_pos[color] == 2:
                count += 1

    print(count)

if __name__ == "__main__":
    main()