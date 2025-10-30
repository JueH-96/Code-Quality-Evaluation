class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        friends = [False] * n
        current = 0
        turn = 1
        
        while not friends[current]:
            friends[current] = True
            current = (current + turn * k) % n
            turn += 1
        
        return [i + 1 for i, friend in enumerate(friends) if not friend]