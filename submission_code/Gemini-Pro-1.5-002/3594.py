class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        
        def check(arr, outlier):
            total_sum = sum(arr)
            for i in range(len(arr) + 1):
                current_sum = 0
                temp = []
                for j in range(len(arr) + 1):
                    if i != j:
                        temp.append(arr[j])

                current_sum = sum(temp[:-1])
                if current_sum == temp[-1]:
                    return True
            return False

        
        potential_outlier1 = nums[-1]
        temp1 = nums[:-1]
        if check(temp1, potential_outlier1):
            return potential_outlier1

        potential_outlier2 = nums[-2]
        temp2 = nums[:-2] + [nums[-1]]
        if check(temp2, potential_outlier2):
            return potential_outlier2
        
        return -1