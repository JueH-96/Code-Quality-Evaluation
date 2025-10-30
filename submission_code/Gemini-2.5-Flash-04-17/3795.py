from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)

        # Helper function to check if the first k queries can make nums zero
        def check(k: int) -> bool:
            # k represents the number of queries from index 0 up to k-1
            if k == 0:
                # If k=0, no queries are applied, so nums must already be all zeros
                return all(x == 0 for x in nums)

            # Calculate the total potential reduction at each index from the first k queries.
            # For a query [l_i, r_i, val_i], we can choose any subset of indices in [l_i, r_i]
            # to decrement by val_i. This means that for any index j within [l_i, r_i],
            # the value val_i is *potentially available* to reduce nums[j].
            # The total potential reduction at index j after applying the first k queries
            # is the sum of val_i for all queries i from 0 to k-1 such that j falls within [l_i, r_i].
            total_reduction_at_index = [0] * n
            for i in range(k):
                l, r, val = queries[i]
                # Apply the potential reduction val to all indices in the query range [l, r]
                for j in range(l, r + 1):
                    total_reduction_at_index[j] += val

            # To make nums[j] zero, the total amount reduced from nums[j] must equal its initial value.
            # Since we can select a subset of indices for each query, and the total potential
            # reduction at index j is the sum of all val_i from queries covering j, we can achieve
            # a reduction of exactly nums[j] if and only if the total potential reduction is
            # greater than or equal to the initial value of nums[j].
            for j in range(n):
                if total_reduction_at_index[j] < nums[j]:
                    return False # Not enough potential reduction at index j

            return True # Sufficient potential reduction at all indices


        # The problem asks for the minimum non-negative value of k.
        # The `check(k)` function is monotonic: if `check(k)` is True, then `check(k+1)` is also True
        # because using more queries can only increase (or keep the same) the total potential reduction
        # at any index. This property allows us to use binary search on k.
        
        # We are searching for the minimum k in the range [0, m] (inclusive) such that check(k) is True.
        
        low = 0
        high = m # The maximum possible k is the total number of queries
        ans = -1 # Initialize result to -1, as required if no such k exists

        # Standard binary search template to find the smallest k satisfying a condition.
        # We search in the range [low, high].
        while low <= high:
            mid = (low + high) // 2
            
            if check(mid):
                # If check(mid) is True, it means using the first `mid` queries is sufficient
                # to potentially make the array zero. This `mid` is a possible answer.
                # We record this potential answer and try to find a smaller k by searching
                # in the left half of the current range [low, mid-1].
                ans = mid
                high = mid - 1
            else:
                # If check(mid) is False, it means using the first `mid` queries is not sufficient.
                # We need more queries, so we must search in the right half of the current range [mid+1, high].
                low = mid + 1

        # After the binary search loop finishes, `ans` will hold the minimum value of `mid`
        # found in the range [0, m] for which `check(mid)` returned True. If `check(mid)`
        # was never true for any `mid` in the range (which implies `check(m)` is False),
        # `ans` will retain its initial value of -1.
        return ans