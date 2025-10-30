class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        total_surplus = 0
        answer = 0
        for limit in usageLimits:
            total_surplus += limit
            while total_surplus >= answer + 1:
                answer += 1
                total_surplus -= answer
        return answer