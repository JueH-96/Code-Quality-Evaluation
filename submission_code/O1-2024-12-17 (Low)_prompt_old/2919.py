class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usage limits in non-decreasing order
        usageLimits.sort()

        # This will keep track of the total credits we have to form groups
        total = 0
        # This will keep track of how many groups we can form so far
        answer = 0

        # Iterate through each usage limit
        for limit in usageLimits:
            total += limit
            # Check if we can form the next group of size (answer + 1).
            # The total number of elements needed to form (answer + 1) groups is
            # 1 + 2 + ... + (answer + 1) = (answer + 1) * (answer + 2) / 2
            # We need total >= that sum to form all those groups.
            if total >= (answer + 1) * (answer + 2) // 2:
                answer += 1
        
        return answer