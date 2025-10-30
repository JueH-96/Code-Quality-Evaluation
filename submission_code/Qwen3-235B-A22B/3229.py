import bisect

class Solution:
    def minimumCost(self, nums):
        def is_palindrome(x):
            if x < 1:
                return False
            s = str(x)
            return s == s[::-1]
        
        def generate_candidates(m):
            s = str(m)
            l = len(s)
            prefix = int(s[:( (l + 1) // 2 )])
            res = []
            for dx in [-1, 0, 1]:
                p = prefix + dx
                if l % 2 == 0:
                    pal = str(p) + str(p)[::-1]
                else:
                    pal = str(p) + str(p)[:-1][::-1]
                res.append(int(pal))
            res.append(10**l + 1)
            res.append(10**l - 1)
            return res
        
        nums.sort()
        n = len(nums)
        m_list = []
        if n % 2 == 0:
            m1 = nums[(n // 2) - 1]
            m2 = nums[n // 2]
            m_list = [m1, m2]
        else:
            m_list = [nums[n // 2]]
        
        candidates = set()
        # Generate candidates within window around medians
        for m in m_list:
            lower = max(1, m - 200)
            upper = m + 200
            for x in range(lower, upper + 1):
                if is_palindrome(x):
                    candidates.add(x)
        
        # Generate helper candidates
        helper_candidates = []
        for m in m_list:
            helper_candidates.extend(generate_candidates(m))
        for y in helper_candidates:
            if 1 <= y < 10**9:
                candidates.add(y)
        
        # Add some edge cases
        edge_candidates = [1, 9, 99, 999, 9999, 99999, 1001, 1111, 121, 2, 101, 111]
        for y in edge_candidates:
            if 1 <= y < 10**9:
                candidates.add(y)
        
        # Compute prefix sums for efficient cost calculation
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_cost = float('inf')
        for y in candidates:
            pos = bisect.bisect_right(nums, y)
            left_sum = prefix[pos]
            right_sum = prefix[-1] - left_sum
            cost = y * pos - left_sum + (right_sum - y * (n - pos))
            if cost < min_cost:
                min_cost = cost
        
        return min_cost