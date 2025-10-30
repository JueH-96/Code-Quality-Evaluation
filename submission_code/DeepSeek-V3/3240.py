class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(n):
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        def get_price(num):
            s = bin(num)[2:][::-1]  # Reverse to get 1-indexed from right
            price = 0
            for i in range(len(s)):
                if (i + 1) % x == 0 and s[i] == '1':
                    price += 1
            return price

        low = 1
        high = 10**18
        best = 0

        while low <= high:
            mid = (low + high) // 2
            total = 0
            for num in range(1, mid + 1):
                total += get_price(num)
                if total > k:
                    break
            if total <= k:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        return best