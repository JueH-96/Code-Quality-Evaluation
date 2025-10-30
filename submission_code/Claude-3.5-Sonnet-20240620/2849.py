class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        
        for i in range(n):
            seen = set()
            imbalance = 0
            max_num = float('-inf')
            min_num = float('inf')
            
            for j in range(i, n):
                num = nums[j]
                if num in seen:
                    total += imbalance
                    continue
                
                if num > max_num:
                    imbalance += 1
                    if num > max_num + 1:
                        imbalance += 1
                    max_num = num
                elif num < min_num:
                    imbalance += 1
                    if num < min_num - 1:
                        imbalance += 1
                    min_num = num
                else:
                    if num + 1 not in seen and num - 1 not in seen:
                        imbalance += 1
                
                if max_num - min_num > 1:
                    imbalance -= 1
                
                seen.add(num)
                total += imbalance
        
        return total