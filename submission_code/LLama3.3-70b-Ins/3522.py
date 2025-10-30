from typing import List

class Solution:
    def get_power(self, subarray: List[int]) -> int:
        """
        Calculate the power of a subarray.
        
        The power of an array is defined as:
        - Its maximum element if all of its elements are consecutive and sorted in ascending order.
        - -1 otherwise.
        
        Args:
        subarray (List[int]): The input subarray.
        
        Returns:
        int: The power of the subarray.
        """
        # Check if the subarray is sorted in ascending order
        if subarray != sorted(subarray):
            return -1
        
        # Check if the subarray elements are consecutive
        if max(subarray) - min(subarray) != len(subarray) - 1:
            return -1
        
        # If the subarray is sorted and consecutive, return its maximum element
        return max(subarray)

    def get_results(self, nums: List[int], k: int) -> List[int]:
        """
        Calculate the power of all subarrays of size k in the given array.
        
        Args:
        nums (List[int]): The input array.
        k (int): The size of the subarrays.
        
        Returns:
        List[int]: A list of powers of all subarrays of size k.
        """
        results = []
        for i in range(len(nums) - k + 1):
            # Extract the subarray of size k
            subarray = nums[i:i+k]
            # Calculate the power of the subarray
            power = self.get_power(subarray)
            # Append the power to the results list
            results.append(power)
        return results

    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        """
        Calculate the power of all subarrays of size k in the given array.
        
        Args:
        nums (List[int]): The input array.
        k (int): The size of the subarrays.
        
        Returns:
        List[int]: A list of powers of all subarrays of size k.
        """
        return self.get_results(nums, k)