# YOUR CODE HERE
def solve():
    h, w = map(int, input().split())
    si, sj = map(int, input().split())
    grid = [input() for _ in range(h)]
    x = input()

    ci, cj = si - 1, sj - 1

    for move in x:
        if move == 'U':
            ni, nj = ci - 1, cj
        elif move == 'D':
            ni, nj = ci + 1, cj
        elif move == 'L':
            ni, nj = ci, cj - 1
        elif move == 'R':
            ni, nj = ci, cj + 1
        
        if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] == '.':
            ci, cj = ni, nj

    print(ci + 1, cj + 1)

solve()