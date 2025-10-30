class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)

        def is_possible(k):
            last_occurrence = {}
            present_indices = set()
            for i in range(k):
                index = changeIndices[i]
                last_occurrence[index] = i + 1
                present_indices.add(index)

            if len(present_indices) < n:
                return False

            for i in range(1, n + 1):
                if i not in last_occurrence:
                    return False

            for i in range(1, n + 1):
                last_time = last_occurrence[i]
                decrement_opportunities = 0
                for s in range(1, last_time):
                    current_index = changeIndices[s - 1]
                    if last_occurrence[current_index] != s:
                        decrement_opportunities += 1

                if decrement_opportunities < nums[i - 1]:
                    return False

            return True

        low = 1
        high = m
        ans = -1

        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        return ans