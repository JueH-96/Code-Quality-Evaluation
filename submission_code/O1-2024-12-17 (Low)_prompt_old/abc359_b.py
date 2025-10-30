def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # positions[color] will store the indices (0-based) where the color appears
    positions = [[] for _ in range(N+1)]

    # Collect positions of each color
    for i, color in enumerate(A):
        positions[color].append(i)

    answer = 0
    # For each color, check if the difference in positions is exactly 2
    # which means exactly one person is between them (e.g., positions c1 and c2 => c2 - c1 == 2)
    for color in range(1, N+1):
        if abs(positions[color][0] - positions[color][1]) == 2:
            answer += 1

    print(answer)

# Let's call solve() to execute
def main():
    solve()

if __name__ == "__main__":
    main()