from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        n = len(pizzas)
        k = n // 4
        
        # Sort pizzas in ascending order
        pizzas.sort()
        
        total_gain = 0
        
        # We have k = n/4 days.
        # On odd days (1, 3, ...), we eat 4 pizzas {W, X, Y, Z} with W<=X<=Y<=Z
        # and gain the weight Z.
        # On even days (2, 4, ...), we eat 4 pizzas {W, X, Y, Z} with W<=X<=Y<=Z
        # and gain the weight Y.
        
        # To maximize total gain, we should aim to make the pizzas contributing to the sum
        # as large as possible. The problem can be optimally solved by a specific
        # grouping of pizzas and assignment of groups to days.
        
        # Consider forming k groups where group i (0-indexed) consists of pizzas
        # {pizzas[3*i], pizzas[3*i+1], pizzas[3*i+2], pizzas[n-1-i]}.
        # When sorted, these pizzas are ordered as w_i=pizzas[3*i], x_i=pizzas[3*i+1],
        # y_i=pizzas[3*i+2], z_i=pizzas[n-1-i]. This grouping is valid because
        # pizzas[3*i+2] <= pizzas[3k-1] and pizzas[n-1-i] >= pizzas[n-k] = pizzas[3k],
        # and 3k-1 < 3k for k >= 1, ensuring pizzas[3*i+2] <= pizzas[n-1-i] in sorted order.
        # Also, the indices used {3i, 3i+1, 3i+2, n-1-i} for i=0...k-1 cover all indices
        # from 0 to n-1 exactly once.
        
        # There are (k+1)//2 odd days and k//2 even days.
        # We assign the (k+1)//2 groups with the largest z_i values to the odd days
        # to maximize the Z contribution. The z_i values across all groups i=0...k-1
        # are pizzas[n-1], pizzas[n-2], ..., pizzas[n-k].
        # The largest (k+1)//2 among these are pizzas[n-1], pizzas[n-2], ..., pizzas[n - ((k+1)//2)].
        # These correspond to group indices i = 0, 1, ..., (k+1)//2 - 1.
        # The gain from these odd days is the sum of their z_i values.
        num_odd_days = (k + 1) // 2
        
        # The sum of Z values from groups assigned to odd days
        # These are pizzas[n-1], pizzas[n-2], ..., pizzas[n - num_odd_days]
        total_gain += sum(pizzas[n - num_odd_days : n])
            
        # The remaining k - num_odd_days = k//2 groups are assigned to even days.
        # These are the groups with indices i = num_odd_days, ..., k-1.
        # For these groups, the gain is y_i = pizzas[3*i + 2].
        # The sum of these Y values is the second part of the total gain.
        for i in range(num_odd_days, k):
            total_gain += pizzas[3 * i + 2]
            
        return total_gain