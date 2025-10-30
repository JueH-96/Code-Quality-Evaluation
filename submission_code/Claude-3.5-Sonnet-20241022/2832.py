class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        positions = {}
        for i, num in enumerate(nums):
            if num in positions:
                positions[num].append(i)
            else:
                positions[num] = [i]
        
        max_len = 1
        for num in positions:
            pos = positions[num]
            if len(pos) == 1:
                continue
                
            left = 0
            right = 0
            curr_len = 1
            gaps = 0
            
            while right < len(pos) - 1:
                gaps += pos[right + 1] - pos[right] - 1
                right += 1
                
                while gaps > k:
                    gaps -= pos[left + 1] - pos[left] - 1
                    left += 1
                    
                curr_len = right - left + 1
                max_len = max(max_len, curr_len)
                
        return max_len