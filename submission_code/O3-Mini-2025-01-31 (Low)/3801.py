class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(num: int) -> bool:
            s = 0
            p = 1
            # calculate sum and product of digits
            while num:
                d = num % 10
                s += d
                p *= d
                num //= 10
            # avoid division by zero, however if s==0 then num was 0, but 0 is not positive.
            return s != 0 and p % s == 0
        
        cnt = 0
        for num in range(l, r+1):
            if is_beautiful(num):
                cnt += 1
        return cnt

# For testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.beautifulNumbers(10, 20))  # Expected output: 2
    # Example 2:
    print(sol.beautifulNumbers(1, 15))   # Expected output: 10