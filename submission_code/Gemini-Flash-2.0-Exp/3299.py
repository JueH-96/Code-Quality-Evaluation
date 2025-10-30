class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        ans = 0
        for num in sorted(counts.keys()):
            if num == 1:
                ans = max(ans, counts[num] - (0 if counts[num] % 2 == 0 else 1))
            else:
                length = 0
                curr = num
                while curr in counts:
                    length += 1
                    if counts[curr] == 0:
                        break
                    counts[curr] -= 1
                    curr *= curr
                ans = max(ans, 2 * (length - 1) + 1)
        
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        if 1 in counts:
            if counts[1] % 2 == 0:
                ans = max(ans, counts[1] - 1)
            else:
                ans = max(ans, counts[1])
        
        return ans