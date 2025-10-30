class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        # Sort the array so that any contiguous segment in this sorted array
        # is a candidate subset to be turned into the same value (its median).
        arr = sorted(nums)
        n = len(arr)
        
        # Prefix sums for fast range-sum queries:
        # S[i] = sum of arr[0] + arr[1] + ... + arr[i-1].
        # Then sum(arr[i..j]) = S[j+1] - S[i].
        S = [0] * (n + 1)
        for i in range(n):
            S[i+1] = S[i] + arr[i]
        
        # A helper function to compute the total cost of converting
        # arr[i..j] all into the median of that segment.
        # Because arr is sorted, the median is arr[m] where m = (i+j)//2.
        # The cost is the sum of absolute differences: sum(|arr[x] - arr[m]|)
        # We can compute it in O(1) using prefix sums.
        def cost(i, j):
            if i > j:
                return 0
            m = (i + j) // 2
            median = arr[m]
            
            # Left side cost = median * number_of_left_elements - sum_of_left_elements
            left_count = m - i + 1
            sum_left = S[m+1] - S[i]  # sum of arr[i..m]
            cost_left = median * left_count - sum_left
            
            # Right side cost = sum_of_right_elements - median * number_of_right_elements
            right_count = j - m
            sum_right = S[j+1] - S[m+1]  # sum of arr[m+1..j]
            cost_right = sum_right - median * right_count
            
            return cost_left + cost_right
        
        # We use a two-pointer (sliding window) approach. "left" and "right"
        # define the current segment [left..right] in the sorted array.
        # We expand "right" and shrink from "left" if the cost is too large.
        left = 0
        answer = 1  # With at least one element, frequency is 1.
        
        for right in range(n):
            # While this window's cost exceeds k, move left forward.
            while left <= right and cost(left, right) > k:
                left += 1
            # Now [left..right] is a valid window if left <= right.
            if left <= right:
                answer = max(answer, right - left + 1)
        
        return answer