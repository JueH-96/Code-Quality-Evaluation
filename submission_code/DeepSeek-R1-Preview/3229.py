class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            s = str(x)
            return s == s[::-1]
        
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        candidates = set()
        
        if n % 2 == 1:
            median = nums_sorted[n // 2]
            for delta in range(-200, 201):
                candidate = median + delta
                if candidate < 1:
                    continue
                candidates.add(candidate)
            s = str(median)
            left = s[:(len(s) + 1) // 2]
            mirror_odd = left + left[:-1][::-1]
            mirror_even = left + left[::-1]
            candidates.add(int(mirror_odd))
            candidates.add(int(mirror_even))
            left_plus = str(int(left) + 1)
            left_minus = str(int(left) - 1)
            if left_minus and int(left_minus) >= 0:
                mirror_plus_odd = left_plus + left_plus[:-1][::-1]
                mirror_plus_even = left_plus + left_plus[::-1]
                mirror_minus_odd = left_minus + left_minus[:-1][::-1]
                mirror_minus_even = left_minus + left_minus[::-1]
                candidates.add(int(mirror_plus_odd))
                candidates.add(int(mirror_plus_even))
                candidates.add(int(mirror_minus_odd))
                candidates.add(int(mirror_minus_even))
        else:
            mid1 = nums_sorted[(n // 2) - 1]
            mid2 = nums_sorted[n // 2]
            for m in [mid1, mid2]:
                for delta in range(-200, 201):
                    candidate = m + delta
                    if candidate < 1:
                        continue
                    candidates.add(candidate)
                s = str(m)
                left = s[:(len(s) + 1) // 2]
                mirror_odd = left + left[:-1][::-1]
                mirror_even = left + left[::-1]
                candidates.add(int(mirror_odd))
                candidates.add(int(mirror_even))
                left_plus = str(int(left) + 1)
                left_minus = str(int(left) - 1)
                if left_minus and int(left_minus) >= 0:
                    mirror_plus_odd = left_plus + left_plus[:-1][::-1]
                    mirror_plus_even = left_plus + left_plus[::-1]
                    mirror_minus_odd = left_minus + left_minus[:-1][::-1]
                    mirror_minus_even = left_minus + left_minus[::-1]
                    candidates.add(int(mirror_plus_odd))
                    candidates.add(int(mirror_plus_even))
                    candidates.add(int(mirror_minus_odd))
                    candidates.add(int(mirror_minus_even))
        
        pal_candidates = [y for y in candidates if is_palindrome(y)]
        if not pal_candidates:
            return 0
        
        min_cost = float('inf')
        for y in pal_candidates:
            cost = sum(abs(x - y) for x in nums_sorted)
            if cost < min_cost:
                min_cost = cost
        
        return min_cost