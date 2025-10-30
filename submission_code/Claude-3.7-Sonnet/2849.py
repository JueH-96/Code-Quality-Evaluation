class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        import bisect
        n = len(nums)
        total_imbalance = 0
        
        for i in range(n):
            sorted_elements = []
            
            for j in range(i, n):
                # Insert the new element at the correct position
                pos = bisect.bisect_left(sorted_elements, nums[j])
                sorted_elements.insert(pos, nums[j])
                
                # Compute the imbalance number
                imbalance = 0
                for k in range(len(sorted_elements) - 1):
                    if sorted_elements[k+1] - sorted_elements[k] > 1:
                        imbalance += 1
                
                total_imbalance += imbalance
        
        return total_imbalance