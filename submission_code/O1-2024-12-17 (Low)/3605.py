class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        # We want to find, for each prime p in nums,
        # the smallest non-negative integer x such that:
        #    x OR (x+1) = p
        # If no such x exists, we set the answer to -1.
        #
        # Since p <= 1000, we can safely brute-force x
        # up to a reasonable limit (e.g., 1024) and pick
        # the smallest x that satisfies the condition.
        
        ans = []
        for p in nums:
            found = False
            best_x = -1
            
            # We'll search x in range [0..1024] (since p <= 1000, this is enough)
            for x in range(1025):
                if (x | (x + 1)) == p:
                    best_x = x
                    found = True
                    break
            
            ans.append(best_x if found else -1)
        
        return ans