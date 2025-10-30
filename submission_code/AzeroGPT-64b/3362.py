from sortedcontainers import SortedList

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = SortedList()
        uniqueness_map = {nums[0]: 1}
        sorted_nums.add(1)
        median = 1

        for i in range(1, n):
            uniqueness_map[nums[i]] = uniqueness_map.get(nums[i], 0) + 1

            if uniqueness_map[nums[i]] == 1:
                sorted_nums.add(len(sorted_nums) + 1)

            queue = SortedList([nums[i]])
            for j in range(i, 0, -1):
                if nums[j - 1] not in queue:
                    queue.add(nums[j - 1])
                    current_uniqueness = len(queue)
                else:
                    current_uniqueness = len(queue)

                if uniqueness_map[nums[j - 1]] == 1:
                    sorted_nums.add(current_uniqueness)
                median = sorted_nums[len(sorted_nums) // 2]

                queue.remove(nums[j])
                uniqueness_map[nums[j]] -= 1

        return median