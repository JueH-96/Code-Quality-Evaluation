from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Explanation:
        # We have n pizzas (n is a multiple of 4) that must be partitioned into m = n/4 groups.
        # In each group of 4 pizzas, if we sort the group as W <= X <= Y <= Z then:
        #   • if the day is odd-numbered we "gain" Z (the largest),
        #   • if the day is even-numbered we "gain" Y (the third largest).
        # We are allowed to partition (and then reassign days arbitrarily) to maximize the total gain.
        #
        # An optimal strategy uses the following greedy idea:
        #   1. We want the very largest pizzas to “count” as gain. Since in an odd day the gain is the group’s maximum,
        #      we assign the highest available pizzas as the “d” (largest) in odd groups.
        #   2. For even groups (where the gain is the third largest, i.e. Y in the sorted quadruple)
        #      we have to “support” a high Y by using one even-higher pizza as a filler (the group’s maximum, which we do not count)
        #      plus two extra (lowest) pizzas to fill the group.
        #
        # Based on this, we can simulate a grouping strategy by first sorting pizzas in descending order.
        #
        # Let m = n / 4 be the total number of groups.
        # The number of odd-day groups is odd_count = (m + 1) // 2 (because day1,3,5,...)
        # and even_day groups count is even_count = m // 2.
        #
        # We simulate the grouping by maintaining two pointers:
        #    - "front" for taking pizzas from the high end,
        #    - "back" for taking pizzas from the low end.
        #
        # For each odd group, we:
        #    • Take the current highest pizza from the front as the candidate d (the group’s maximum) and add it to our answer.
        #    • Remove three pizzas from the back (the smallest available) to complete the group.
        #
        # For each even group, we must form a group so that after sorting the group the third smallest (i.e. second from the lower
        # two from the top two) is as high as possible. The structure is:
        #    Group (in descending order): d, c, b, a.
        #    • We “sacrifice” the top available pizza as d (which is not added to the answer) so that we can use the next pizza, c, as the contribution.
        #    • Then we remove two pizzas from the back to complete the group.
        #
        # In code, we keep moving our "front" pointer and "back" pointer accordingly.
        #
        # The correctness of this greedy approach follows from the fact that we want to “preserve” our very large pizzas
        # for the contribution positions (either as direct gain in odd groups or as support for a high candidate in even groups).
        #
        # Time complexity is O(n log n) due to sorting.
        
        pizzas.sort(reverse=True)
        n = len(pizzas)
        m = n // 4  # total number of days (groups)
        # Determine how many groups (days) are odd and even.
        odd_count = (m + 1) // 2   # day 1, 3, 5, ...
        even_count = m // 2
        
        res = 0
        front = 0            # pointer for the large end
        back = n - 1         # pointer for the small end
        
        # Process odd-day groups:
        for _ in range(odd_count):
            # In an odd-day group, the gain is the group's maximum.
            res += pizzas[front]
            front += 1         # used one pizza from the front for candidate (Z)
            back -= 3          # remove three smallest pizzas as fillers
        
        # Process even-day groups:
        for _ in range(even_count):
            # For an even-day group, we want a high candidate for the third largest.
            # We first use one high pizza as filler (position d, not added to answer)
            front += 1         # skip one high pizza (filler d)
            # Next pizza becomes candidate c (gain to add)
            res += pizzas[front]
            front += 1         # use candidate c
            back -= 2          # remove two fillers from the back
        
        return res


# Below is code to run some tests.
if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    pizzas1 = [1,2,3,4,5,6,7,8]
    print(sol.maxWeight(pizzas1))  # Expected output: 14
    # Explanation:
    # Day 1 (odd): Group: [2,3,5,8] -> gain = 8.
    # Day 2 (even): Group: [1,4,6,7] -> sorted = [1,4,6,7] -> gain = 6.
    # Total = 8 + 6 = 14.
    
    # Example 2:
    pizzas2 = [2,1,1,1,1,1,1,1]
    print(sol.maxWeight(pizzas2))  # Expected output: 3
    # Explanation:
    # Day 1 (odd): Group: [1,1,1,2] -> gain = 2.
    # Day 2 (even): Group: [1,1,1,1] -> gain = 1.
    # Total = 2 + 1 = 3.