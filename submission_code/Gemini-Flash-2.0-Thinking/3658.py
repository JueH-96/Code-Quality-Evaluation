class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)

        def check(diff):
            def can_satisfy(arr):
                temp_nums = list(arr)
                neg_one_indices = [i for i, x in enumerate(temp_nums) if x == -1]

                if not neg_one_indices:
                    for i in range(len(temp_nums) - 1):
                        if abs(temp_nums[i] - temp_nums[i + 1]) > diff:
                            return False
                    return True

                num_neg_ones = len(neg_one_indices)

                def solve(index, current_arr):
                    if index == num_neg_ones:
                        for i in range(len(current_arr) - 1):
                            if abs(current_arr[i] - current_arr[i + 1]) > diff:
                                return False
                        return True

                    neg_one_index = neg_one_indices[index]

                    lower_bound = 1
                    upper_bound = 10**9 + 7

                    # Constraints from left neighbor
                    if neg_one_index > 0 and current_arr[neg_one_index - 1] != -1:
                        lower_bound = max(lower_bound, current_arr[neg_one_index - 1] - diff)
                        upper_bound = min(upper_bound, current_arr[neg_one_index - 1] + diff)

                    # Constraints from right neighbor
                    if neg_one_index < len(current_arr) - 1 and current_arr[neg_one_index + 1] != -1:
                        lower_bound = max(lower_bound, current_arr[neg_one_index + 1] - diff)
                        upper_bound = min(upper_bound, current_arr[neg_one_index + 1] + diff)

                    original_value = current_arr[neg_one_index]

                    for val in range(1, 201): # Optimization: Check a small range
                        current_arr[neg_one_index] = val
                        if solve(index + 1, current_arr):
                            current_arr[neg_one_index] = original_value
                            return True
                    current_arr[neg_one_index] = original_value
                    return False

                return solve(0, temp_nums)

            first_neg = -1
            for i, x in enumerate(nums):
                if x == -1:
                    first_neg = i
                    break

            if first_neg == -1:
                for i in range(n - 1):
                    if abs(nums[i] - nums[i + 1]) > diff:
                        return False
                return True

            possible_values = set()
            for val in range(1, 201): # Optimization: Check a small range
                possible_values.add(val)

            import itertools
            neg_indices = [i for i, x in enumerate(nums) if x == -1]
            num_neg = len(neg_indices)

            for combo in itertools.product(possible_values, repeat=num_neg):
                temp_nums = list(nums)
                for i, val in enumerate(combo):
                    temp_nums[neg_indices[i]] = val

                is_valid = True
                for i in range(n - 1):
                    if abs(temp_nums[i] - temp_nums[i + 1]) > diff:
                        is_valid = False
                        break
                if is_valid:
                    return True
            return False

        left, right = 0, 10**9
        ans = right

        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans