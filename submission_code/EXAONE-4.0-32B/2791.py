class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()
        current = 0
        visited.add(current)
        turn = 1
        
        while True:
            next_index = (current + turn * k) % n
            if next_index in visited:
                break
            visited.add(next_index)
            current = next_index
            turn += 1
        
        losers = []
        for i in range(n):
            if i not in visited:
                losers.append(i + 1)
                
        return losers