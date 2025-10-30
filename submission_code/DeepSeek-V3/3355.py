class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Precompute the total sum of the array
        total = sum(possible)
        # Initialize the sum for Alice
        alice_sum = 0
        # Iterate through each possible split
        for i in range(n-1):
            if possible[i] == 1:
                alice_sum += 1
            else:
                alice_sum -= 1
            # Bob's sum is total - alice_sum
            bob_sum = total - alice_sum
            # Since Bob plays optimally, he will maximize his points
            # So, for the levels he plays, he will choose to play all possible levels
            # But since the levels are fixed, we can directly compute his sum
            # Wait, no. The levels are fixed in order, so Bob will play the remaining levels in order
            # So, his sum is the sum of the remaining levels
            # So, the condition is alice_sum > bob_sum
            if alice_sum > bob_sum:
                return i + 1
        return -1