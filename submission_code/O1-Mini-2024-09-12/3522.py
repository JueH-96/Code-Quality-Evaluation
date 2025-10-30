class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            window = nums[i:i + k]
            is_consecutive = True
            for j in range(1, k):
                if window[j] != window[j - 1] + 1:
                    is_consecutive = False
                    break
            if is_consecutive:
                power = window[-1]  # Since the window is sorted, the last element is the maximum
            else:
                power = -1
            results.append(power)
        return results