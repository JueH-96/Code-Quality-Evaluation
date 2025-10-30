class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        max_sum = 0
        current_sum = 0
        freq = {}

        # Process the first window of size k
        for i in range(k):
            element = nums[i]
            current_sum += element
            freq[element] = freq.get(element, 0) + 1

        # Check the first window
        if len(freq) >= m:
            max_sum = current_sum

        # Slide the window
        # The loop variable `i` is the index of the new element entering the window.
        # The window being considered after the update ends at index `i`
        # and starts at index `i - k + 1`.
        for i in range(k, n):
            # Add the new element entering the window (at index i)
            entering_element = nums[i]
            current_sum += entering_element
            freq[entering_element] = freq.get(entering_element, 0) + 1

            # Remove the old element leaving the window (at index i-k)
            leaving_element = nums[i - k]
            current_sum -= leaving_element
            freq[leaving_element] -= 1
            if freq[leaving_element] == 0:
                del freq[leaving_element] # Remove element from map if count is zero

            # Check the current window (which ends at index i)
            # len(freq) gives the number of distinct elements in the window nums[i-k+1 : i+1]
            if len(freq) >= m:
                max_sum = max(max_sum, current_sum)

        return max_sum