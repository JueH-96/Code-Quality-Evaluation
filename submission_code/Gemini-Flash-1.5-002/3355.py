class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        ans = -1
        for i in range(1, n):
            alice_score = 0
            bob_score = 0
            alice_possible = possible[:i]
            bob_possible = possible[i:]
            
            alice_ones = sum(alice_possible)
            alice_zeros = len(alice_possible) - alice_ones
            alice_score = alice_ones - alice_zeros

            bob_ones = sum(bob_possible)
            bob_zeros = len(bob_possible) - bob_ones
            bob_score = bob_ones - bob_zeros

            if alice_score > bob_score:
                ans = i
                break
        return ans