class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * (n + 1)   # Index 0 unused, friends from 1..n
        visited[1] = True            # 1st friend starts with the ball
        current = 1                  # Current holder of the ball
        step = 1                     # Multiplier for k
        
        while True:
            # Determine next friend (modular arithmetic, using 1-based indexing)
            next_friend = (current - 1 + step * k) % n + 1
            # If friend already got the ball, game ends
            if visited[next_friend]:
                break
            visited[next_friend] = True
            current = next_friend
            step += 1
        
        # Losers are those who never received the ball
        return [i for i in range(1, n + 1) if not visited[i]]