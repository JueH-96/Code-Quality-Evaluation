class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = [False] * n
        current = 0  # Start at friend 1 (0-indexed)
        received[current] = True
        
        turn = 1
        while True:
            # Pass the ball turn*k steps clockwise
            current = (current + turn * k) % n
            
            # If this friend already received the ball, game ends
            if received[current]:
                break
            
            received[current] = True
            turn += 1
        
        # Find all friends who never received the ball (1-indexed)
        return [i + 1 for i in range(n) if not received[i]]