class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        from collections import Counter
        
        n = len(nums)
        answer = []
        
        for start in range(n - k + 1):
            subarray = nums[start:start + k]
            freq = Counter(subarray)
            
            # Sort by frequency desc, then value desc
            # items() -> (num, freq), so sort by (-freq, -num)
            sorted_freq = sorted(freq.items(), key=lambda item: (item[1], item[0]), reverse=True)
            
            # If we have fewer than x distinct elements, sum the subarray
            if len(freq) <= x:
                answer.append(sum(subarray))
            else:
                # Otherwise, pick top x distinct elements
                kept_values = set([val for val, _ in sorted_freq[:x]])
                
                # Sum only those top x elements from the subarray
                sub_sum = sum([val for val in subarray if val in kept_values])
                answer.append(sub_sum)
        
        return answer