class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = set()
        current = 1
        turn = 1
        
        while True:
            if current in received:
                break
            received.add(current)
            current = ((current - 1 + turn * k) % n) + 1
            turn += 1
        
        losers = [i for i in range(1, n+1) if i not in received]
        return losers