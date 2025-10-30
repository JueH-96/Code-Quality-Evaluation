class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            sub_array = nums[i:i + k]
            is_consecutive = True
            is_sorted = True
            for j in range(1, k):
                if sub_array[j] != sub_array[j - 1] + 1:
                    is_consecutive = False
                    break
            if not is_consecutive:
                results.append(-1)
                continue

            for j in range(1,k):
                if sub_array[j] < sub_array[j-1]:
                    is_sorted = False
                    break
            if not is_sorted:
                results.append(-1)
                continue

            results.append(max(sub_array))
        return results