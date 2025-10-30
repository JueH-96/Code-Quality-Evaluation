class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Precompute the total points for all levels
        total = sum(1 if x == 1 else -1 for x in possible)
        # Initialize Alice's points and Bob's points
        alice_points = 0
        # Iterate through each level to find the minimum k
        for k in range(1, n):
            # Update Alice's points
            alice_points += 1 if possible[k-1] == 1 else -1
            # Bob's points is total - alice_points
            bob_points = total - alice_points
            if alice_points > bob_points:
                return k
        return -1