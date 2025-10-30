class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            sub_array = nums[i:i+k]
            is_sorted = True
            is_consecutive = True
            if k > 1:
                for j in range(k - 1):
                    if sub_array[j] > sub_array[j+1]:
                        is_sorted = False
                        break
                if is_sorted:
                    for j in range(k - 1):
                        if sub_array[j+1] != sub_array[j] + 1:
                            is_consecutive = False
                            break
                else:
                    is_consecutive = False
            elif k == 1:
                is_sorted = True
                is_consecutive = True

            if is_sorted and is_consecutive:
                results.append(max(sub_array))
            else:
                results.append(-1)
        return results