from collections import deque

def solve_coloring_problem(N, K):
    def is_on_line(point, player_position):
        return point == player_position or (N % 2 == 0 and point == (player_position + N // 2) % N)
    
    def is_valid_move(board, i, player_position):
        if is_on_line(i, player_position) and board[i] == '0':
            return True
        reflection = (2 * player_position - i) % N
        return board[i] == '0' and board[reflection] == '1'
    
    # The state is represented by (board, player)
    # The board is a string for hashability
    queue = deque([('0' * N, 0)])  # 0 for Alice, 1 for Bob
    visited = {('0' * N, 0)}
    
    while queue:
        board, player = queue.popleft()
        
        # Check if all points are black
        if all(c == '1' for c in board):
            return True
        
        # Players: 0 for Alice, 1 for Bob
        player_position = 0 if player == 0 else K
        
        found_valid_move = False
        for i in range(N):
            if board[i] == '0' and is_valid_move(board, i, player_position):
                # Color the point and switch player
                new_board = board[:i] + '1' + board[i+1:]
                
                new_state = (new_board, 1 - player)
                if new_state not in visited:
                    queue.append(new_state)
                    visited.add(new_state)
                    found_valid_move = True
        
        # If no valid move found for this player, the game ends for this sequence
        if not found_valid_move:
            continue
    
    # No valid sequence found
    return False

def main():
    T = int(input())
    results = []
    
    for _ in range(T):
        N, K = map(int, input().split())
        if solve_coloring_problem(N, K):
            results.append("Yes")
        else:
            results.append("No")
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()