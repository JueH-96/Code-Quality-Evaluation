class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        """
        We can only swap adjacent elements if they have the same number of set bits.
        Effectively, this means elements are 'locked' into segments of consecutive
        identical bit-counts. Within each segment, we can reorder arbitrarily, but
        we cannot move elements across segments.

        Steps:
        1. Compute the bit counts for each element.
        2. Partition the array into consecutive segments where the bit count remains the same.
        3. Within each segment, sort the elements.
        4. If the resulting array (after sorting each segment) is in sorted order, return True,
           otherwise return False.
        """
        # Function to compute the number of set bits:
        def bit_count(x: int) -> int:
            return bin(x).count('1')
        
        n = len(nums)
        if n <= 1:
            return True
        
        # Precompute bit counts for each element
        bits = [bit_count(x) for x in nums]
        
        # Identify consecutive segments of same bit counts
        segments = []
        start = 0
        for i in range(1, n):
            if bits[i] != bits[i - 1]:
                segments.append((start, i - 1))
                start = i
        segments.append((start, n - 1))
        
        # Create a copy of the array and sort each segment in-place
        arr = nums[:]
        for left, right in segments:
            arr[left:right + 1] = sorted(arr[left:right + 1])
        
        # Check if the modified array is globally sorted
        return all(arr[i] <= arr[i+1] for i in range(n-1))