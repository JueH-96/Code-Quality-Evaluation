class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(n):
            even_count = 0
            odd_count = 0
            while n > 0:
                digit = n % 10
                if digit % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
                n //= 10
            return even_count == odd_count
        
        beautiful_count = 0
        for num in range(low, high + 1):
            if num % k == 0 and is_beautiful(num):
                beautiful_count += 1
        return beautiful_count

# Example usage:
# sol = Solution()
# print(sol.numberOfBeautifulIntegers(10, 20, 3))  # Output: 2
# print(sol.numberOfBeautifulIntegers(1, 10, 1))   # Output: 1
# print(sol.numberOfBeautifulIntegers(5, 5, 2))    # Output: 0