class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()
        current = 0
        step = k
        turn = 1
        
        while current not in visited:
            visited.add(current)
            current = (current + turn * k) % n
            turn += 1
        
        return [i + 1 for i in range(n) if i not in visited]