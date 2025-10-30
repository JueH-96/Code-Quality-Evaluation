def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Record the two positions for each color.
    positions = [[] for _ in range(N + 1)]
    for idx, color in enumerate(A):
        positions[color].append(idx)

    # Count how many colors have exactly one person between their two occurrences.
    count = 0
    for color in range(1, N + 1):
        p1, p2 = positions[color]
        if p2 - p1 == 2:
            count += 1

    print(count)

if __name__ == "__main__":
    main()