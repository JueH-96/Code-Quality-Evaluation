def solution(nums: List[int], k: int) -> int:
    if not nums:
        return 0
    n = len(nums)
    if k == 1:
        current_sum = 0
        max_sum = float('-inf')
        for num in nums:
            current_sum += num
            if current_sum > max_sum:
                max_sum = current_sum
            if current_sum < 0:
                current_sum = 0
        return max_sum if max_sum != float('-inf') else 0
    else:
        dp_prev = [-float('inf')] * k
        if k == 1:
            return 0
        dp_prev[1] = nums[0]
        for i in range(1, n):
            new_dp = [-float('inf')] * k
            for r in range(k):
                prev_r = (r - 1) % k
                if dp_prev[prev_r] != -float('inf'):
                    option1 = dp_prev[prev_r] + nums[i]
                else:
                    option1 = -float('inf')
                if r == 0:
                    option2 = -float('inf')
                else:
                    option2 = nums[i] if r == 1 else -float('inf')
                new_dp[r] = max(option1, option2)
            dp_prev = new_dp
        max_sum = dp_prev[0] if dp_prev[0] != -float('inf') else 0
        return max_sum if max_sum != -float('inf') else 0