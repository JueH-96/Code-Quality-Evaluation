from typing import List
import math

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # The key idea:
        # We need to partition the sorted pizzas (of total count n = 4*k) into k groups of 4.
        # In each group, after sorting it as [W, X, Y, Z], if the group is consumed on:
        #   - an odd day, you gain Z (its maximum)
        #   - an even day, you gain Y (its 3rd smallest, or 2nd largest)
        #
        # Since you can reorder the groups arbitrarily (i.e. choose which groups are odd and which even),
        # you want to “upgrade” some groups from a base gain of Y to a gain of Z.
        # In other words, if every group contributed Y, the total gain would be sum(Y for each group).
        # Then for some groups you can "upgrade" an even day group (which grants Y) to an odd day group (which
        # grants Z). The additional bonus for such an upgrade is (Z - Y).
        #
        # Because you have exactly ceil(k/2) odd days available, you can choose at most m = ceil(k/2) groups to upgrade.
        # To maximize the sum, you want to pick those groups with the largest differences (Z - Y).
        #
        # Now it turns out that an optimal partition is obtained by first sorting the pizzas.
        # Then, think of splitting the sorted array into three segments:
        #   • The first 2*k pizzas are used as "fillers" (they serve as the lower two numbers in every group).
        #   • The next k pizzas (indices 2*k to 3*k - 1) play the role of the potential even‐day gain (Y).
        #   • The last k pizzas (indices 3*k to 4*k - 1) are the candidates for Z.
        #
        # In any valid group, since the filler numbers are the smallest, we can always set:
        #   group = [filler1, filler2, candidate_c, candidate_d] (sorted as [filler1, filler2, candidate_c, candidate_d])
        # so that if the group is “even” it yields candidate_c and if “odd” it yields candidate_d.
        #
        # The baseline total gain if all groups were consumed on even days is:
        #     baseline = sum(candidate_c)  (i.e. sum of pizzas in positions 2*k to 3*k - 1)
        #
        # Then, you can upgrade m = ceil(k/2) groups to odd days.
        # You are free to pair candidate_c and candidate_d arbitrarily among groups.
        # To maximize the additional bonus, you want to match the smallest candidate_c with the largest candidate_d.
        #
        # So the optimal additional bonus is:
        #     bonus = (sum of m largest candidate_d) - (sum of m smallest candidate_c)
        #
        # And the answer is:
        #     total = baseline + bonus
        #
        # Note: k = n // 4.
        
        n = len(pizzas)
        # Total groups count. n is a multiple of 4.
        k = n // 4
        
        # Sort the pizzas in non-decreasing order.
        pizzas.sort()
        
        # Partition the array:
        #   fillers: pizzas[0 : 2*k]  (not used in score)
        #   candidate_c (for even day groups): pizzas[2*k : 3*k]
        #   candidate_d (for odd day groups): pizzas[3*k : 4*k]
        candidate_c = pizzas[2 * k : 3 * k]
        candidate_d = pizzas[3 * k : 4 * k]
        
        # Baseline gain: if all groups gave Y (i.e., candidate_c)
        baseline = sum(candidate_c)
        
        # Number of groups that must be upgraded to odd days.
        m = (k + 1) // 2  # this is equivalent to ceil(k / 2)
        
        # For the upgrade, we want to choose m pairs:
        #    Pair the m smallest candidate_c (which are candidate_c[0:m])
        #    with the m largest candidate_d (which are candidate_d[-m:])
        bonus = sum(candidate_d[-m:]) - sum(candidate_c[:m])
        
        return baseline + bonus