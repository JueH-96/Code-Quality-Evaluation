class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        """
        We want to find the largest subset of nums that can be turned into the same value
        using at most numOperations single-use modifications, where each chosen element can
        be shifted by up to ±k.

        Key insight:
        - If we sort nums, then any subset we make identical must lie (after shifting) in
          some interval of length <= 2k. That is, for a chosen target X in that interval,
          each element in the subset must satisfy |nums[i] - X| <= k.
        - This is equivalent to saying, in sorted order, the difference between the minimum
          and maximum in the chosen window is at most 2k (to have a non-empty intersection
          of [nums[i]-k, nums[i]+k]).
        - Even if the intersection condition is satisfied, we still need to ensure we do
          not exceed numOperations distinct changes to make them all one value. If in a
          chosen window of length w, the most frequent element appears freq times, we need
          at most w - freq changes to unify the window to that element. If w - freq <=
          numOperations, the whole window can be made identical.
        
        We'll use a two-pointer (sliding window) approach over the sorted array:
        1. Sort nums -> arr.
        2. Use two pointers l and r to keep track of a valid window where arr[r] - arr[l] <= 2k.
        3. Maintain a frequency map of elements in the current window, plus a "frequency of
           frequencies" map to track the maximum frequency in O(1) time.
        4. Expand r, and while the window is invalid (arr[r] - arr[l] > 2k), move l up.
        5. At each valid window, if (window_size - current_max_freq) <= numOperations,
           we can unify that window. Track the best (max) window size.

        Time Complexity: O(n), because each element is added and removed from the window
        at most once, and we maintain the maximum frequency in O(1) for each update.
        """
        import collections
        
        arr = sorted(nums)
        freq = collections.defaultdict(int)       # Maps element -> how many times it appears in window
        countFreq = collections.defaultdict(int)  # Maps "frequency" -> how many elements have this frequency
        maxFreq = 0                               # Tracks the maximum frequency in the current window
        
        def add_value(x):
            nonlocal maxFreq
            old_count = freq[x]
            new_count = old_count + 1
            freq[x] = new_count
            
            if old_count > 0:
                countFreq[old_count] -= 1
                if countFreq[old_count] == 0:
                    del countFreq[old_count]
            
            countFreq[new_count] += 1
            maxFreq = max(maxFreq, new_count)
        
        def remove_value(x):
            nonlocal maxFreq
            old_count = freq[x]
            new_count = old_count - 1
            freq[x] = new_count
            
            countFreq[old_count] -= 1
            if countFreq[old_count] == 0:
                del countFreq[old_count]
            
            if new_count > 0:
                countFreq[new_count] += 1
            
            # If we removed the last element having old_count == maxFreq, we decrement maxFreq
            if old_count == maxFreq and old_count not in countFreq:
                maxFreq -= 1
        
        l = 0
        best = 1
        for r in range(len(arr)):
            # Add the new element
            add_value(arr[r])
            
            # Shrink from left if window is invalid for intersection (need arr[r] - arr[l] <= 2k)
            while arr[r] - arr[l] > 2*k:
                remove_value(arr[l])
                l += 1
            
            # Now [l..r] is a valid window in terms of intersection
            window_size = (r - l + 1)
            # Cost to unify to the most frequent element is window_size - maxFreq
            cost = window_size - maxFreq
            if cost <= numOperations:
                best = max(best, window_size)
        
        return best