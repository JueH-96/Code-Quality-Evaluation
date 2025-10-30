class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        zeros = s.count('0')
        ones = n - zeros
        if abs(zeros - ones) <= numOps:
            return 1
        
        def check(length):
            ops_needed = 0
            for i in range(n - length + 1):
                substring = s[i:i+length]
                zeros_count = substring.count('0')
                ones_count = length - zeros_count
                
                if zeros_count > ones_count:
                    ops_needed = ones_count
                else:
                    ops_needed = zeros_count
                
                if ops_needed <= numOps:
                    return True
            return False

        left, right = 1, n
        ans = n
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans