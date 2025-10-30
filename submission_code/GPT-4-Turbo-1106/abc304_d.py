def read_ints():
    return list(map(int, input().split()))

def main():
    W, H = read_ints()
    N = int(input())
    strawberries = [read_ints() for _ in range(N)]
    A = int(input())
    x_cuts = read_ints()
    B = int(input())
    y_cuts = read_ints()

    # Add the borders as cuts
    x_cuts = [0] + x_cuts + [W]
    y_cuts = [0] + y_cuts + [H]

    # Create a 2D grid to count strawberries in each piece
    grid = [[0] * (A + 2) for _ in range(B + 2)]

    # Place strawberries in the grid
    for x, y in strawberries:
        # Find the piece where the strawberry belongs
        x_piece = next(i for i in range(A + 1) if x_cuts[i] < x <= x_cuts[i + 1])
        y_piece = next(i for i in range(B + 1) if y_cuts[i] < y <= y_cuts[i + 1])
        grid[y_piece][x_piece] += 1

    # Find the minimum and maximum number of strawberries in the pieces
    min_strawberries = float('inf')
    max_strawberries = float('-inf')
    for row in grid:
        for cell in row:
            min_strawberries = min(min_strawberries, cell)
            max_strawberries = max(max_strawberries, cell)

    # Print the result
    print(min_strawberries, max_strawberries)

if __name__ == "__main__":
    main()