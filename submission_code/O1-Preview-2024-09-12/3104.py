class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums_sorted = sorted(nums)
        nums_set = set(nums)
        ans = 0
        count = 0
        prev = -1
        nums_sorted.append(n)  # Append n to handle the last interval

        i = 0
        while i <= n:
            current = nums_sorted[i]
            # Check for k between prev+1 and current-1
            while prev +1 < current:
                prev +=1
                if count == prev and prev not in nums_set:
                    ans +=1
            if i < n:
                count +=1  # Include current student
                prev = current
            i +=1
        
        return ans