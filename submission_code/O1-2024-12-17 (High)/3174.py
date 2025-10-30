class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        changes = 0
        # Process the string in pairs of length 2
        for i in range(0, n, 2):
            pair = s[i:i+2]
            # Cost to make this pair "00" is the number of '1's in the pair
            cost_make_zero = pair.count('1')
            # Cost to make this pair "11" is the number of '0's in the pair
            cost_make_one = pair.count('0')
            # Take the minimal cost
            changes += min(cost_make_zero, cost_make_one)
        return changes