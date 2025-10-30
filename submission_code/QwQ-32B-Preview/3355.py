class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        # Compute total sum
        total_sum = 0
        for p in possible:
            if p == 1:
                total_sum += 1
            else:
                total_sum -= 1
        
        # Iterate through k from 0 to n-2
        alice_sum = 0
        n = len(possible)
        for k in range(n-1):
            if possible[k] == 1:
                alice_sum += 1
            else:
                alice_sum -= 1
            if 2 * alice_sum > total_sum:
                return k + 1
        return -1