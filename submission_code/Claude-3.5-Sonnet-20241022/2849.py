class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        
        for i in range(n):
            seen = set()
            imbalance = 0
            seen.add(nums[i])
            
            for j in range(i, n):
                curr = nums[j]
                
                if curr not in seen:
                    # Check if curr-1 and curr+1 exist in seen
                    has_prev = curr-1 in seen
                    has_next = curr+1 in seen
                    
                    if has_prev and has_next:
                        imbalance -= 1
                    elif not has_prev and not has_next:
                        imbalance += 1
                        
                    seen.add(curr)
                
                total += imbalance
                
        return total