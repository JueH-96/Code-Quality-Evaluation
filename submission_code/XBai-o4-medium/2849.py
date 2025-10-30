class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        max_num = n  # since nums[i] is between 1 and n
        
        for i in range(n):
            present = [False] * (max_num + 2)  # 0 to max_num + 1
            current_imbalance = 0
            for j in range(i, n):
                x = nums[j]
                if present[x]:
                    total += current_imbalance
                    continue
                present[x] = True
                
                # Find left (max element less than x)
                left = None
                k = x - 1
                while k >= 1:
                    if present[k]:
                        left = k
                        break
                    k -= 1
                
                # Find right (min element greater than x)
                right = None
                k = x + 1
                while k <= max_num:
                    if present[k]:
                        right = k
                        break
                    k += 1
                
                # Calculate original contribution
                original = 0
                if left is not None and right is not None:
                    if (right - left) > 1:
                        original = 1
                    current_imbalance -= original
                
                # Calculate new contribution
                new_contribution = 0
                if left is not None:
                    if (x - left) > 1:
                        new_contribution += 1
                if right is not None:
                    if (right - x) > 1:
                        new_contribution += 1
                current_imbalance += new_contribution
                
                total += current_imbalance
        
        return total