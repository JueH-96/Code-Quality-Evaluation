class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = set([0])  # friend 1 has the ball at the beginning
        current_pos = 0
        step = 1
        while True:
            next_pos = (current_pos + step * k) % n
            if next_pos in received:
                break
            received.add(next_pos)
            current_pos = next_pos
            step += 1
        return [i + 1 for i in range(n) if i not in received]