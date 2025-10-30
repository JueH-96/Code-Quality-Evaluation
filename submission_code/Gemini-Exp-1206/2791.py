class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * n
        curr = 0
        step = 1
        while not visited[curr]:
            visited[curr] = True
            curr = (curr + step * k) % n
            step += 1
        
        losers = []
        for i in range(n):
            if not visited[i]:
                losers.append(i + 1)
        
        return losers