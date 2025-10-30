import sys
sys.setrecursionlimit(1000000)

def main():
    A = []
    for _ in range(3):
        data = sys.stdin.readline().split()
        A.append(list(map(int, data)))
    
    board = [['.'] * 3 for _ in range(3)]
    
    def check_win(b, i, j, c):
        if b[i][0] == c and b[i][1] == c and b[i][2] == c:
            return True
        if b[0][j] == c and b[1][j] == c and b[2][j] == c:
            return True
        if i == j:
            if b[0][0] == c and b[1][1] == c and b[2][2] == c:
                return True
        if i + j == 2:
            if b[0][2] == c and b[1][1] == c and b[2][0] == c:
                return True
        return False

    memo = {}
    
    def dfs(b):
        n_moves = 0
        score_t = 0
        score_a = 0
        available_moves = []
        for i in range(3):
            for j in range(3):
                cell = b[i][j]
                if cell == '.':
                    available_moves.append((i, j))
                elif cell == 'R':
                    n_moves += 1
                    score_t += A[i][j]
                elif cell == 'B':
                    n_moves += 1
                    score_a += A[i][j]
                    
        if n_moves == 9:
            if score_t > score_a:
                return 1
            else:
                return -1
        
        turn = n_moves % 2
        key = tuple(tuple(row) for row in b)
        if key in memo:
            return memo[key]
            
        if turn == 0:
            best = -10**18
        else:
            best = 10**18
            
        for (i, j) in available_moves:
            new_b = [list(row) for row in b]
            if turn == 0:
                new_b[i][j] = 'R'
                c = 'R'
            else:
                new_b[i][j] = 'B'
                c = 'B'
                
            if check_win(new_b, i, j, c):
                if turn == 0:
                    res = 1
                else:
                    res = -1
            else:
                res = dfs(new_b)
                
            if turn == 0:
                if res > best:
                    best = res
                if res == 1:
                    memo[key] = 1
                    return 1
            else:
                if res < best:
                    best = res
                if res == -1:
                    memo[key] = -1
                    return -1
                    
        memo[key] = best
        return best
        
    result = dfs(board)
    if result == 1:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()