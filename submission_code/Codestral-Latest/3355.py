class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        alice_score = 0
        bob_score = 0

        # Calculate the total score for Bob if he plays all levels
        for i in range(n):
            if possible[i] == 1:
                bob_score += 1
            else:
                bob_score -= 1

        # Check each level Alice can play up to
        for i in range(n):
            if possible[i] == 1:
                alice_score += 1
            else:
                alice_score -= 1

            # Calculate Bob's score for the remaining levels
            remaining_bob_score = bob_score
            for j in range(i + 1, n):
                if possible[j] == 1:
                    remaining_bob_score -= 1
                else:
                    remaining_bob_score += 1

            # Check if Alice has more points than Bob
            if alice_score > remaining_bob_score:
                return i + 1

        return -1