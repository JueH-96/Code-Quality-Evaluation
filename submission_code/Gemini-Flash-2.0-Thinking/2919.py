class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)

        def can_form_groups(num_groups):
            required_elements = num_groups * (num_groups + 1) // 2
            if sum(usageLimits) < required_elements:
                return False

            sorted_limits = sorted(usageLimits)
            current_usage = 0
            elements_used = 0

            for group_size in range(1, num_groups + 1):
                elements_needed_for_group = group_size

                available_elements = 0
                for limit in sorted_limits:
                    if limit > 0:
                        available_elements += 1

                if available_elements < elements_needed_for_group:
                    return False

                temp_limits = list(sorted_limits)
                elements_in_group = 0
                group_numbers = []

                for _ in range(elements_needed_for_group):
                    found = False
                    for i in range(len(temp_limits)):
                        if temp_limits[i] > 0:
                            group_numbers.append(i)
                            temp_limits[i] -= 1
                            found = True
                            break
                    if not found:
                        return False

                for idx in group_numbers:
                    sorted_limits[idx] -= 1

            return True

        left, right = 0, n
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if can_form_groups(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans