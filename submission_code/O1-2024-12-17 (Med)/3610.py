class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        from collections import Counter
        
        n = len(nums)
        result = []
        
        for i in range(n - k + 1):
            sub = nums[i:i + k]
            freq = Counter(sub)
            
            # If there are fewer than x distinct elements, just sum the subarray
            if len(freq) <= x:
                result.append(sum(sub))
            else:
                # Sort by frequency (descending), then by value (descending)
                # freq.items() -> (element, count)
                sorted_items = sorted(freq.items(), key=lambda p: (p[1], p[0]), reverse=True)
                
                # Take the top x items
                top_x = sorted_items[:x]
                
                # Sum up freq_of_key * key for those top x items
                x_sum = sum(key * count for key, count in top_x)
                result.append(x_sum)
        
        return result