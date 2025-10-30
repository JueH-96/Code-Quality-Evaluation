class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        total_adjacent = 0
        ans = []
        
        for idx, color in queries:
            old_color = nums[idx]
            if old_color == color:
                ans.append(total_adjacent)
                continue
                
            left = idx - 1
            right = idx + 1
            
            if old_color != 0:
                if left >= 0 and nums[left] == old_color:
                    total_adjacent -= 1
                if right < n and nums[right] == old_color:
                    total_adjacent -= 1
                    
            nums[idx] = color
            
            if left >= 0 and nums[left] == color:
                total_adjacent += 1
            if right < n and nums[right] == color:
                total_adjacent += 1
                
            ans.append(total_adjacent)
            
        return ans