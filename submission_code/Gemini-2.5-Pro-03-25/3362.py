import collections
from typing import List

class Solution:
    """
    Finds the median of the uniqueness array of nums.
    The uniqueness array is the sorted array containing the number of distinct elements 
    for all possible subarrays of the input array `nums`.
    The median is defined as the middle element of the sorted array. If the array has an
    even number of elements, the median is the smaller of the two middle elements.
    This corresponds to the element at 1-based index ceil(N/2) where N is the total 
    number of subarrays.

    The approach uses binary search on the possible values of the median (number of distinct elements),
    which range from 1 to n (length of nums). For a candidate median value `k`, we efficiently count
    how many subarrays have at most `k` distinct elements using a sliding window technique.
    Let this count be `count_le(k)`.
    The total number of subarrays is `total_subarrays = n * (n + 1) // 2`.
    The rank of the median element (1-based index) is `K = (total_subarrays + 1) // 2`.
    We are looking for the smallest integer `ans` such that `count_le(ans) >= K`.
    The binary search finds this `ans`.

    Time Complexity: O(N log N), where N is the length of `nums`. The binary search performs log N iterations.
                     Each iteration calls `count_le(k)` which takes O(N) time using the sliding window.
    Space Complexity: O(N) in the worst case for the frequency map used in the sliding window,
                      if all elements in `nums` are distinct.
    """
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        """
        :param nums: List[int] - The input integer array. Constraints: 1 <= nums.length <= 10^5, 1 <= nums[i] <= 10^5.
        :return: int - The median of the uniqueness array.
        """
        n = len(nums)
        
        # Calculate the total number of subarrays.
        # This can be up to approximately 5 * 10^9 for n=10^5. Python's arbitrary precision integers handle this.
        total_subarrays = n * (n + 1) // 2
        
        # Calculate the 1-based rank K of the median element in the sorted uniqueness array.
        # The median definition corresponds to the element at rank ceil(N/2).
        # The formula (N + 1) // 2 correctly calculates ceil(N/2) for integer N.
        K = (total_subarrays + 1) // 2 

        def count_le(k: int) -> int:
            """
            Helper function to count the number of subarrays `nums[i..j]`
            such that the number of distinct elements in the subarray is at most `k`.
            Uses a sliding window approach with two pointers `i` (left) and `j` (right).
            
            :param k: int - The maximum allowed number of distinct elements.
            :return: int - The count of subarrays satisfying the condition distinct(nums[i..j]) <= k.
            """
            count = 0  # Initialize count of valid subarrays
            # Use collections.Counter for efficient frequency tracking of elements in the window.
            freq = collections.Counter() 
            i = 0  # Left pointer of the sliding window
            
            # Iterate with the right pointer `j` through the array
            for j in range(n):
                # Add the element nums[j] into the window and update its frequency
                freq[nums[j]] += 1
                
                # The number of distinct elements currently in the window `nums[i..j]` is `len(freq)`.
                # If this count exceeds `k`, we need to shrink the window from the left by moving `i` forward.
                while len(freq) > k:
                    # Decrement frequency of the element at the left boundary `nums[i]`
                    freq[nums[i]] -= 1
                    # If the frequency drops to 0, remove the element key from the Counter.
                    # This ensures `len(freq)` accurately reflects the count of distinct elements present in the window.
                    if freq[nums[i]] == 0:
                        del freq[nums[i]]
                    # Move the left pointer forward, effectively shrinking the window
                    i += 1
                
                # After the potential shrinking, the window `nums[i..j]` satisfies the condition:
                # the number of distinct elements is at most `k`.
                # Any subarray ending at `j` and starting at index `p` where `i <= p <= j`
                # is fully contained within `nums[i..j]`. Therefore, such subarrays also have at most `k` distinct elements.
                # The number of such valid starting indices `p` is `(j - i + 1)`.
                # Add this number to the total count.
                count += (j - i + 1)
            
            # Return the total count of subarrays with at most k distinct elements.
            return count

        # Binary search for the median value. The value represents the number of distinct elements.
        # The possible range for the number of distinct elements in any subarray is [1, n].
        low = 1   # Minimum possible distinct count is 1
        high = n  # Maximum possible distinct count is n
        ans = n   # Initialize the answer to the maximum possible value

        # The binary search aims to find the smallest integer `ans` such that `count_le(ans) >= K`.
        # This `ans` corresponds to the K-th smallest value in the uniqueness array, which is the median.
        while low <= high:
            # Calculate the middle value `mid` for the current search range.
            # Using `low + (high - low) // 2` avoids potential overflow issues in languages with fixed-size integers.
            mid = low + (high - low) // 2 
            
            # Calculate how many subarrays have at most `mid` distinct elements.
            num_subarrays_le_mid = count_le(mid)

            if num_subarrays_le_mid >= K:
                # If the count `num_subarrays_le_mid` is at least `K`, it means `mid` could be the median.
                # The true median might be `mid` or potentially a smaller value.
                # So, we record `mid` as a candidate answer and narrow the search to the lower half: `[low, mid - 1]`.
                ans = mid
                high = mid - 1 
            else:
                # If the count `num_subarrays_le_mid` is less than `K`, it means `mid` is too small.
                # The median must have more than `mid` distinct elements.
                # We narrow the search to the upper half: `[mid + 1, high]`.
                low = mid + 1
        
        # The loop terminates when `low > high`. `ans` holds the smallest value `k`
        # for which `count_le(k) >= K`. This is the desired median value.
        return ans