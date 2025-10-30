class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()
        current = 0
        visited.add(current)
        turn = 1
        
        while True:
            steps = turn * k
            next_pos = (current + steps) % n
            if next_pos in visited:
                break
            visited.add(next_pos)
            current = next_pos
            turn += 1
        
        losers = [i + 1 for i in range(n) if i not in visited]
        return sorted(losers)