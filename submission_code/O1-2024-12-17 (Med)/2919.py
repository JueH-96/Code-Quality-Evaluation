class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usage limits in non-decreasing order
        usageLimits.sort()
        
        total = 0          # This will accumulate the total usage we have so far
        answer = 0         # Number of groups formed
        
        # Iterate over the sorted usage limits
        for limit in usageLimits:
            total += limit
            # While we can afford to create a group of size (answer+1),
            # use up that many "slots" and form the group
            while total >= (answer + 1):
                answer += 1
                total -= answer
        
        return answer