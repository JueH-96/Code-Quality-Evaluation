class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * n
        current = 0
        visited[current] = True
        turn = 1
        
        while True:
            next_pos = (current + turn * k) % n
            if visited[next_pos]:
                break
            visited[next_pos] = True
            current = next_pos
            turn += 1
        
        losers = []
        for i in range(n):
            if not visited[i]:
                losers.append(i + 1)
        return losers