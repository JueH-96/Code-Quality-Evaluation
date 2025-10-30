class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        changes = 0
        for i in range(0, n, 2):
            sub = s[i:i+2]
            count_0 = sub.count('0')
            count_1 = sub.count('1')

            # Cost to make the substring "00"
            cost_00 = count_1

            # Cost to make the substring "11"
            cost_11 = count_0

            changes += min(cost_00, cost_11)

        return changes