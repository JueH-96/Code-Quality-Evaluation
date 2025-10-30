class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received_ball = [False] * n
        received_ball[0] = True
        current_position = 0
        i = 1
        
        while True:
            next_position = (current_position + i * k) % n
            if received_ball[next_position]:
                break
            received_ball[next_position] = True
            current_position = next_position
            i += 1
        
        return [i + 1 for i in range(n) if not received_ball[i]]