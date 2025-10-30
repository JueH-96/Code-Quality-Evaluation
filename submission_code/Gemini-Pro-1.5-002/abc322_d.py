# YOUR CODE HERE
def solve():
    polyominoes = []
    for _ in range(3):
        polyomino = []
        for _ in range(4):
            polyomino.append(input())
        polyominoes.append(polyomino)

    def rotate(polyomino):
        rotated = [['.' for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                rotated[j][3 - i] = polyomino[i][j]
        return ["".join(row) for row in rotated]

    def count_cells(polyomino):
        count = 0
        for row in polyomino:
            for cell in row:
                if cell == '#':
                    count += 1
        return count

    cell_counts = [count_cells(p) for p in polyominoes]
    if sum(cell_counts) != 16:
        print("No")
        return

    def check(arrangement):
        grid = [['.' for _ in range(4)] for _ in range(4)]
        for polyomino in arrangement:
            for r in range(4):
                for c in range(4):
                    if polyomino[r][c] == '#':
                        if grid[r][c] == '#':
                            return False
                        grid[r][c] = '#'
        for r in range(4):
            for c in range(4):
                if grid[r][c] == '.':
                    return False
        return True
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                p1_rotations = [polyominoes[0]]
                for _ in range(i):
                    p1_rotations.append(rotate(p1_rotations[-1]))
                p2_rotations = [polyominoes[1]]
                for _ in range(j):
                    p2_rotations.append(rotate(p2_rotations[-1]))
                p3_rotations = [polyominoes[2]]
                for _ in range(k):
                    p3_rotations.append(rotate(p3_rotations[-1]))

                if check([p1_rotations[-1], p2_rotations[-1], p3_rotations[-1]]):
                    print("Yes")
                    return

    print("No")

solve()