def get_position(num, N):
    row = (num - 1) // N
    col = (num - 1) % N
    return row, col

def check_bingo(marked, N):
    # Check rows
    for i in range(N):
        if all(marked[i][j] for j in range(N)):
            return True
            
    # Check columns
    for j in range(N):
        if all(marked[i][j] for i in range(N)):
            return True
    
    # Check main diagonal
    if all(marked[i][i] for i in range(N)):
        return True
        
    # Check anti diagonal
    if all(marked[i][N-1-i] for i in range(N)):
        return True
            
    return False

N, T = map(int, input().split())
A = list(map(int, input().split()))

marked = [[False]*N for _ in range(N)]

for turn in range(T):
    num = A[turn]
    row, col = get_position(num, N)
    marked[row][col] = True
    
    if check_bingo(marked, N):
        print(turn + 1)
        exit()

print(-1)