class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        nums_list = list(nums) # Convert to list for modification

        for i in range(n - 2):
            if nums_list[i] == 0:
                operations += 1
                for j in range(i, i + 3):
                    nums_list[j] = 1 - nums_list[j]

        last_three = nums_list[n - 3:]
        if all(x == 1 for x in last_three):
            return operations
        elif all(x == 0 for x in last_three):
            return operations + 1
        else:
            return -1

def flip_subarray(arr, start_index):
    for i in range(start_index, start_index + 3):
        arr[i] = 1 - arr[i]
    return arr