class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        """
        We want to perform at most numOperations distinct index operations on nums,
        where each chosen index i can have nums[i] adjusted by any integer in [-k, k].
        
        Our goal is to maximize the frequency of some value in the final array.
        
        --------------------------------------------------
        KEY IDEAS:
        
        1) If we choose a target T that is NOT originally in the array:
           - No element starts out equal to T, so converting an element x to T costs 1 operation
             as long as |x - T| <= k.
           - Hence we can only convert those x for which x is in [T - k, T + k].
           - Because we have at most numOperations operations in total (and each index can be chosen at
             most once), the maximum count we can get is min(numOperations, number_of_x_in_that_range).
           - To find the largest number_of_x_in_that_range for any T, note that [T - k, T + k] is an
             interval of length 2k. So effectively we want to place an interval of length 2k in the
             sorted array to cover as many elements as possible (then we convert up to numOperations
             of them). The resulting frequency = min(numOperations, coverage).
           - If k==0, no element can be changed to a "new" value, so this case yields 0.
           
        2) If we choose a target T that IS originally in the array (say T = v):
           - All elements already equal to v need 0 operations.
           - Any other element x can become v if |x - v| <= k (i.e. x in [v-k, v+k]) and we choose
             that index. We can do this at most numOperations times.
           - Let freq_in_range = number of array elements x with x in [v-k, v+k].
             Among these, c[v] are already v (no cost). The rest (freq_in_range - c[v]) need 1 operation each.
             So if freq_in_range - c[v] <= numOperations, we can make them all v ⇒ frequency = freq_in_range.
             Otherwise we can only convert numOperations of them ⇒ frequency = c[v] + numOperations.
             Hence final frequency for v = min(freq_in_range, c[v] + numOperations).
           
        We compute BOTH:
          - candidate1 = the best possible if T is not in the array (sliding window of width ≤ 2k).
          - candidate2 = the best possible if T is some value v in the array (enumerate distinct v).
        The answer is max(candidate1, candidate2).
        
        TIME COMPLEXITY:
        - Sorting nums: O(n log n).
        - candidate1 via two-pointer sliding window: O(n).
        - candidate2 by distinct-value enumeration + binary search: O(d log n), where d ≤ n is the
          number of distinct values. Overall O(n log n) is feasible for n up to 10^5.
        """
        import bisect
        
        n = len(nums)
        nums.sort()
        
        # ---------------------------
        # 1) Case: T not in the array
        #    We find the largest number of elements covered by an interval of length 2k,
        #    then the frequency is min(numOperations, that coverage).
        #    If k == 0, no element can become a "new" value, so candidate1 = 0.
        # ---------------------------
        candidate1 = 0
        if k > 0:
            left = 0
            for right in range(n):
                while nums[right] - nums[left] > 2*k:
                    left += 1
                coverage = right - left + 1  # how many elements fit within a window of size ≤ 2k
                candidate1 = max(candidate1, min(coverage, numOperations))
        else:
            # k==0 => cannot shift any element to a new value that's different
            candidate1 = 0
        
        # ---------------------------
        # 2) Case: T in the array
        #    For each distinct value v, find how many elements lie in [v-k, v+k].
        #    Let freq_in_range = count of such elements. Let c[v] = number of occurrences of v.
        #    Then freq_for_v = min(freq_in_range, c[v] + numOperations).
        #    We'll take the maximum over all v.
        # ---------------------------
        
        # Build frequency map of values
        from collections import Counter
        count_map = Counter(nums)  # v -> frequency
        distinct_vals = sorted(count_map.keys())
        
        candidate2 = 0
        
        # We'll do a small helper to get the count of x in [L, R] via binary search
        # freq_in_range = rightIndex - leftIndex + 1
        #  leftIndex = bisect_left(nums, L)
        #  rightIndex = bisect_right(nums, R) - 1
        
        for v in distinct_vals:
            L = v - k
            R = v + k
            left_index = bisect.bisect_left(nums, L)
            right_index = bisect.bisect_right(nums, R) - 1
            if right_index >= left_index:
                freq_in_range = (right_index - left_index + 1)
            else:
                freq_in_range = 0
            
            c_v = count_map[v]  # number of v in entire array (which also lie in that range)
            
            # If we can transform all freq_in_range - c_v (provided it doesn't exceed numOperations),
            # we get freq_in_range. Otherwise we get c[v] + numOperations.
            candidate2 = max(candidate2, min(freq_in_range, c_v + numOperations))
        
        # Final answer
        return max(candidate1, candidate2)