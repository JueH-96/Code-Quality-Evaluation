class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        from sortedcontainers import SortedList
        n = len(nums)
        total_imbalance = 0
        
        # Iterate over all possible starting points of subarrays
        for start in range(n):
            sorted_subarray = SortedList()
            # Iterate over all possible ending points of subarrays
            for end in range(start, n):
                sorted_subarray.add(nums[end])
                imbalance_count = 0
                
                # Calculate the imbalance number for the current subarray
                for i in range(len(sorted_subarray) - 1):
                    if sorted_subarray[i + 1] - sorted_subarray[i] > 1:
                        imbalance_count += 1
                
                # Add the imbalance number of the current subarray to the total
                total_imbalance += imbalance_count
        
        return total_imbalance