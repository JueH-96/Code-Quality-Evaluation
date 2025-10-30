def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Dictionary to store the positions of each color.
    positions = {}
    for idx, color in enumerate(A):
        if color not in positions:
            positions[color] = []
        positions[color].append(idx)

    answer = 0
    # Check for each color if the positions differ by exactly 2.
    for color in range(1, N + 1):
        pos = positions[color]
        if abs(pos[0] - pos[1]) == 2:
            answer += 1

    print(answer)

# Do not forget to call main function
if __name__ == "__main__":
    main()