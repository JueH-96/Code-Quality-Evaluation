class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            s = str(x)
            return s == s[::-1]

        def next_pal(n):
            if n < 0:
                return None
            s = str(n)
            length = len(s)
            left = s[:length//2]
            right = s[length//2:]
            candidate = left + left[::-1]
            if candidate > n:
                return int(candidate)
            else:
                new_left = str(int(left)) + '1'
                candidate = new_left + new_left[::-1]
                if len(candidate) != length:
                    return 10**length + 1
                else:
                    if int(candidate) > n:
                        return int(candidate)
                    else:
                        return 10**length + 1

        def prev_pal(n):
            if n < 0:
                return None
            s = str(n)
            length = len(s)
            left = s[:length//2]
            right = s[length//2:]
            candidate = left + right[::-1]
            if int(candidate) < n:
                return int(candidate)
            else:
                new_left = str(int(left)) - '1'
                if new_left < '0':
                    new_left = '9' * (len(left) - 1)
                    if len(new_left) < len(left):
                        new_left = '9' * len(left)
                    candidate = new_left + new_left[::-1]
                else:
                    candidate = new_left + new_left[::-1]
                if int(candidate) < n:
                    return int(candidate)
                else:
                    return 10**(length - 1) - 1

        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        if n % 2 == 1:
            median = sorted_nums[n // 2]
        else:
            median = sorted_nums[n // 2 - 1]

        candidates = []
        if is_palindrome(median):
            candidates.append(median)
        next_p = next_pal(median)
        if next_p is not None:
            candidates.append(next_p)
        prev_p = prev_pal(median)
        if prev_p is not None:
            candidates.append(prev_p)

        min_cost = float('inf')
        for y in candidates:
            cost = 0
            for num in nums:
                cost += abs(num - y)
            if cost < min_cost:
                min_cost = cost
        return min_cost