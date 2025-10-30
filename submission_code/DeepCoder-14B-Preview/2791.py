class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        if n == 0:
            return []
        
        visited = set()
        current = 1
        visited.add(current)
        i = 1
        
        while True:
            steps = i * k
            next_pos = (current - 1 + steps) % n + 1
            if next_pos in visited:
                break
            visited.add(next_pos)
            current = next_pos
            i += 1
        
        losers = [x for x in range(1, n + 1) if x not in visited]
        return sorted(losers)