class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        
        q = [(x, y, "Alice")]
        visited = set()
        while q:
            curr_x, curr_y, player = q.pop(0)
            if (curr_x, curr_y) in visited:
                continue
            visited.add((curr_x, curr_y))

            can_move = False
            for i in range(min(curr_x, 2) + 1):
                remaining = 115 - i * 75
                if remaining >= 0 and remaining % 10 == 0 and remaining // 10 <= curr_y:
                    next_x = curr_x - i
                    next_y = curr_y - remaining // 10
                    
                    can_win = True
                    for j in range(min(next_x, 2) + 1):
                        remaining2 = 115 - j * 75
                        if remaining2 >= 0 and remaining2 % 10 == 0 and remaining2 // 10 <= next_y:
                            can_win = False
                            break
                    if can_win:
                        return "Bob" if player == "Alice" else "Alice"

                    
                    next_player = "Bob" if player == "Alice" else "Alice"
                    q.append((next_x, next_y, next_player))
                    can_move = True

            if not can_move:
                return "Bob" if player == "Alice" else "Alice"
        return "Alice"