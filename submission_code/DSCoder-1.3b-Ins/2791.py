class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()
        current = 1
        losers = []

        while current not in visited and current <= n:
            visited.add(current)
            current = (current + k - 1) % n + 1

        for i in range(1, n + 1):
            if i not in visited:
                losers.append(i)

        return losers