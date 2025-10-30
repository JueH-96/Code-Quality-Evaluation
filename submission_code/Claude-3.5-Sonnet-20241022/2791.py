class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()
        curr = 1
        step = 1
        
        while curr not in visited:
            visited.add(curr)
            curr = ((curr - 1 + step * k) % n) + 1
            step += 1
            
        losers = []
        for i in range(1, n+1):
            if i not in visited:
                losers.append(i)
                
        return losers