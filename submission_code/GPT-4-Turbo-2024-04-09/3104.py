class Solution:
    def countWays(self, nums: List[int]) -> int:
        from collections import Counter
        
        n = len(nums)
        count = Counter(nums)
        
        # Calculate the prefix sum of counts
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + count[i]
        
        # Calculate the number of valid k (number of students selected)
        valid_k = 0
        for k in range(n + 1):
            all_happy = True
            for i in range(n):
                if (k > i and k <= prefix[i + 1]) or (k < i and k >= prefix[i]):
                    continue
                else:
                    all_happy = False
                    break
            if all_happy:
                valid_k += 1
        
        return valid_k