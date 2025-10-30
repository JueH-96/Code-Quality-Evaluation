class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        global_max = 0
        n = len(nums)
        for j in range(1, n - 1):
            i_list = nums[:j]
            a_b = [x - nums[j] for x in i_list]
            max_a_b = max(a_b) if a_b else 0
            min_a_b = min(a_b) if a_b else 0
            k_list = nums[j+1:]
            max_c = max(k_list) if k_list else 0
            min_c = min(k_list) if k_list else 0
            option1 = max_a_b * max_c
            option2 = min_a_b * min_c
            current_max = max(option1, option2)
            if current_max > global_max:
                global_max = current_max
        return global_max if global_max > 0 else 0