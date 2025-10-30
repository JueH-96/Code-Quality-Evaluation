class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        current = 0
        visited = {current}
        i = 1
        while True:
            next_pos = (current + i * k) % n
            if next_pos in visited:
                break
            visited.add(next_pos)
            current = next_pos
            i += 1
        all_friends = set(range(1, n + 1))
        visited_friends = {x + 1 for x in visited}
        losers = sorted(all_friends - visited_friends)
        return losers