class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        freq = [0] * 24
        for hour in hours:
            rem = hour % 24
            freq[rem] += 1
        count = 0
        # Pairs where both remainders are 0
        count += (freq[0] * (freq[0] - 1)) // 2
        # Pairs where both remainders are 12
        count += (freq[12] * (freq[12] - 1)) // 2
        # Pairs where remainders sum to 24 and are different
        for r in range(1, 12):
            count += freq[r] * freq[24 - r]
        return count