class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            sub_array = nums[i:i+k]
            is_sorted = True
            for j in range(k - 1):
                if sub_array[j+1] < sub_array[j]:
                    is_sorted = False
                    break
            if not is_sorted:
                results.append(-1)
            else:
                max_val = max(sub_array)
                min_val = min(sub_array)
                if max_val - min_val == k - 1:
                    results.append(max_val)
                else:
                    results.append(-1)
        return results