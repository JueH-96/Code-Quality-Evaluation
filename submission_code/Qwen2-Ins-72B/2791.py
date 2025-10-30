class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        seen = set()
        friend, step = 0, 1
        while friend not in seen:
            seen.add(friend)
            friend = (friend + step * k) % n
            step += 1
        return [i + 1 for i in range(n) if i not in seen]