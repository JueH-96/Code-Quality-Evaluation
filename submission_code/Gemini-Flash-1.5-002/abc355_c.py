# YOUR CODE HERE
n, t = map(int, input().split())
a = list(map(int, input().split()))

grid = [[0] * n for _ in range(n)]
marked = [[False] * n for _ in range(n)]

def check_bingo():
    # Check rows
    for i in range(n):
        if all(marked[i]):
            return True
    # Check columns
    for j in range(n):
        if all(marked[i][j] for i in range(n)):
            return True
    # Check diagonals
    if all(marked[i][i] for i in range(n)):
        return True
    if all(marked[i][n - 1 - i] for i in range(n)):
        return True
    return False

for turn in range(t):
    num = a[turn]
    row = (num - 1) // n
    col = (num - 1) % n
    marked[row][col] = True
    if check_bingo():
        print(turn + 1)
        break
else:
    print(-1)