from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        """
        Let val[i] be the score obtained on level i :
            val[i] =  1  if possible[i] == 1   (level is clearable)
            val[i] = -1  if possible[i] == 0   (level is impossible)

        Alice plays the prefix 0 .. k   (k ≤ n-2 because Bob must play ≥ 1 level)
        Bob   plays the suffix k+1 .. n-1

        Alice’s score   = prefix_sum[k]
        Bob’s   score   = total_sum - prefix_sum[k]

        We need  prefix_sum[k] > total_sum - prefix_sum[k]
        ⇔ 2 * prefix_sum[k] > total_sum
        ⇔ prefix_sum[k]   > total_sum / 2

        Therefore we scan the array once, maintain the running prefix sum
        and return the first position k (converted to number of levels = k+1)
        for which the inequality holds.  If no such k exists, return -1.
        """
        n = len(possible)
        # score of each level (+1 for 1, -1 for 0)
        values = [1 if x else -1 for x in possible]

        total_sum = sum(values)
        prefix_sum = 0

        # k ranges from 0 to n-2 so that Bob plays at least one level
        for k in range(n - 1):
            prefix_sum += values[k]
            if prefix_sum * 2 > total_sum:
                return k + 1      # number of levels Alice plays
        return -1