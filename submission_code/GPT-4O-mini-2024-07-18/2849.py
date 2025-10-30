class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        
        for start in range(n):
            subarray = []
            for end in range(start, n):
                subarray.append(nums[end])
                sorted_subarray = sorted(subarray)
                imbalance_count = 0
                
                for i in range(len(sorted_subarray) - 1):
                    if sorted_subarray[i + 1] - sorted_subarray[i] > 1:
                        imbalance_count += 1
                
                total_imbalance += imbalance_count
        
        return total_imbalance