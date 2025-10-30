class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        answers = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                peak_count = 0
                sub_array = nums[l : r + 1]
                if len(sub_array) >= 3:
                    for i in range(1, len(sub_array) - 1):
                        if sub_array[i] > sub_array[i - 1] and sub_array[i] > sub_array[i + 1]:
                            peak_count += 1
                answers.append(peak_count)
            elif query[0] == 2:
                index, val = query[1], query[2]
                nums[index] = val
        return answers