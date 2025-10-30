class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * (n + 1)
        current = 1
        count = 1
        visited[current] = True
        while True:
            next_friend = (current + count * k) % n
            if next_friend == 0:
                next_friend = n
            if visited[next_friend]:
                break
            visited[next_friend] = True
            current = next_friend
            count += 1
        losers = []
        for i in range(1, n + 1):
            if not visited[i]:
                losers.append(i)
        return losers