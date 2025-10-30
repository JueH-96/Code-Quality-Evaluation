class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        """
        We want to maximize how many elements in nums can become the same value T
        if, for up to numOperations distinct indices, we can add any integer in [-k, k].
        
        For each element nums[i], it can become any integer in [ nums[i] - k, nums[i] + k ].
        Thus, to see if nums[i] can become T, we need |nums[i] - T| <= k.
        
        We do not need to transform elements that are already T; they cost 0 operations.
        Elements that are not T (but can become T) each cost 1 operation, and we can only
        afford at most numOperations in total. We also cannot reuse an index more than once.
        
        Strategy:
          1. Let min_val = min(nums), max_val = max(nums).
          2. The only relevant T values lie within [min_val - k, max_val + k].
          3. Count how many nums[i] fall in [T - k, T + k] using a frequency array and prefix sums.
             - Then among those, freq[T] are already T (cost 0).
             - We can use up to numOperations operations to convert others in the set to T.
          4. The maximum frequency for T is freq[T] + min(numOperations, count_in_range - freq[T]).
          5. Take the maximum over all valid T.
        
        This runs in O(n + R) time, where R ~ (max_val - min_val + 2k + 1) which can be at most ~300,000.
        This is feasible in Python if implemented efficiently.
        """
        import sys
        
        # Quick edge case: if no operations allowed, just return the most frequent number in nums
        if numOperations == 0:
            from collections import Counter
            return max(Counter(nums).values())
        
        min_val = min(nums)
        max_val = max(nums)
        
        # If k = 0, effectively we can't change the value (only add 0).
        # The answer is just the max frequency of the original array.
        if k == 0:
            from collections import Counter
            return max(Counter(nums).values())
        
        # Frequency array must cover [min_val - k ... max_val + k].
        # That range can be large but is at most 1 <= nums[i] <= 1e5 and k <= 1e5 => up to ~200k range around the array values.
        left_T = min_val - k
        right_T = max_val + k
        
        # To map T -> index, shift all T by (-left_T)
        shift = -left_T  # so that T + shift >= 0
        size = right_T - left_T + 1  # total length of freq array
        
        # Build freq array of original values
        # Each nums[i] goes to freq[nums[i] + shift]
        freq = [0] * (size + 1)  # +1 so we can build prefix sums easily
        for x in nums:
            freq[x + shift] += 1
        
        # Build prefix sums
        for i in range(1, size):
            freq[i] += freq[i - 1]
        
        # Helper to get prefix sum in [L, R], inclusive
        # freq array now is prefix-like: freq[i] holds sum up to index i
        # We'll rewrite freq to be prefix array so freq[i] = sum of original freq up to i
        # so after the for-loop above, freq[i] is the sum from freq[0]..freq[i].
        
        # Now freq[i] = count of all original elements mapped to indices <= i
        # "count_in_range" = freq[R] - freq[L - 1] if L > 0, else freq[R].
        
        def prefix_sum(idx):
            if idx < 0:
                return 0
            if idx >= size:
                idx = size - 1
            return freq[idx]
        
        # We'll iterate T from left_T to right_T
        # For each T:
        #   T_idx = T + shift
        #   freqT = how many were originally T? = prefix_sum(T_idx) - prefix_sum(T_idx - 1).
        #   left_idx = T - k + shift, right_idx = T + k + shift
        #   count_in_range = prefix_sum(right_idx) - prefix_sum(left_idx - 1)
        #   best_for_T = freqT + min(numOperations, count_in_range - freqT)
        
        best = 1  # at least 1, because we can always have frequency 1
        prev_val = -1  # to store freq of T if needed, but we'll just compute on the fly
        
        # For speed, we'll avoid calling prefix_sum repeatedly for freqT
        # Instead, freq(i) = prefix_sum(i) - prefix_sum(i-1).
        
        for T in range(left_T, right_T + 1):
            T_idx = T + shift
            
            # freqT:
            if 0 <= T_idx < size:
                freqT = prefix_sum(T_idx) - prefix_sum(T_idx - 1)
            else:
                # T out of freq array range => freqT=0
                freqT = 0
            
            # We want sum of freq in [T-k, T+k]
            left_idx = (T - k) + shift
            right_idx = (T + k) + shift
            count_in_range = prefix_sum(right_idx) - prefix_sum(left_idx - 1)
            
            can_transform = min(numOperations, count_in_range - freqT)
            best_for_T = freqT + can_transform
            if best_for_T > best:
                best = best_for_T
        
        return best