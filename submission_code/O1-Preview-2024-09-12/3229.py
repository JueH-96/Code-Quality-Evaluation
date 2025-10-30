class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        from bisect import bisect_right

        nums.sort()
        n = len(nums)
        prefix = [0] * n
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] + nums[i]
        
        palindromes = []
        max_num = 10**9
        # Generate palindromic numbers less than 1e9
        # Half lengths from 1 to 5 (since max length is 9)
        for hlen in range(1, 6):
            start = 10**(hlen-1) if hlen > 1 else 1
            end = 10**hlen
            for num in range(start, end):
                # Odd length palindrome
                s = str(num)
                pal = int(s + s[-2::-1])
                if pal < max_num:
                    palindromes.append(pal)
                else:
                    break
                # Even length palindrome
                pal = int(s + s[::-1])
                if pal < max_num:
                    palindromes.append(pal)
                else:
                    break
        palindromes = sorted(set(palindromes))
        min_total_cost = float('inf')
        for y in palindromes:
            # Find the index k where nums[k-1] <= y < nums[k]
            k = bisect_right(nums, y)
            # Compute cost
            if k > 0:
                left_cost = k * y - prefix[k-1]
            else:
                left_cost = 0
            if k < n:
                right_cost = (prefix[n-1] - (prefix[k-1] if k > 0 else 0)) - (n - k) * y
            else:
                right_cost = 0
            total_cost = left_cost + right_cost
            if total_cost < min_total_cost:
                min_total_cost = total_cost
        return min_total_cost