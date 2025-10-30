class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()  # Track which friends have received the ball
        
        current_friend = 1  # Start with friend 1
        turn = 1
        
        while current_friend not in visited:
            visited.add(current_friend)
            steps = turn * k
            current_friend = ((current_friend - 1 + steps) % n) + 1  # Calculate next friend (1-indexed)
            turn += 1
        
        # Return friends who never received the ball, in ascending order
        losers = [i for i in range(1, n+1) if i not in visited]
        return losers