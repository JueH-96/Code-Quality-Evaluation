class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_len = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                sub_array = nums[i:j+1]
                counts = {}
                for num in sub_array:
                    counts[num] = counts.get(num, 0) + 1
                
                deletions = 0
                most_frequent = 0
                for count in counts.values():
                    if count > most_frequent:
                        most_frequent = count

                deletions = len(sub_array) - most_frequent
                if deletions <=k:
                    max_len = max(max_len, len(sub_array))

        return max_len