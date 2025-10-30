def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    arr = [int(next(it)) for _ in range(2 * n)]

    # Record the two positions for each color
    positions = [[] for _ in range(n + 1)]
    for idx, color in enumerate(arr):
        positions[color].append(idx)

    # Count how many colors have exactly one person between their two occurrences
    answer = 0
    for color in range(1, n + 1):
        p1, p2 = positions[color]
        if abs(p2 - p1) == 2:
            answer += 1

    print(answer)

if __name__ == "__main__":
    main()