class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received_ball = set()
        current_friend = 1
        received_ball.add(current_friend)
        turn = 1
        
        while True:
            next_friend = ((current_friend - 1 + turn * k) % n) + 1
            if next_friend in received_ball:
                break
            received_ball.add(next_friend)
            current_friend = next_friend
            turn += 1
        
        losers = []
        for i in range(1, n + 1):
            if i not in received_ball:
                losers.append(i)
        
        return losers