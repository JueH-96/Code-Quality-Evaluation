class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        from collections import Counter
        
        n = len(nums)
        answer = []
        
        for start in range(n - k + 1):
            subarray = nums[start:start + k]
            freq = Counter(subarray)
            
            # Sort by frequency desc, then value desc
            items = sorted(freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
            
            # Pick the top x elements (if there are fewer than x, take them all)
            top_x = items[:x]
            
            # Sum of subarray after keeping only top x distinct elements
            sub_sum = sum(val * count for val, count in top_x)
            answer.append(sub_sum)
        
        return answer