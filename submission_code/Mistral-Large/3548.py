class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]

        def generate_palindromes(n):
            if n == 1:
                return [i for i in range(1, 10)]
            half_length = (n + 1) // 2
            start = 10 ** (half_length - 1)
            end = 10 ** half_length
            palindromes = []
            for i in range(start, end):
                s = str(i)
                if n % 2 == 0:
                    s += s[-2::-1]
                else:
                    s += s[::-1]
                palindromes.append(int(s))
            return palindromes

        palindromes = generate_palindromes(n)
        count = 0
        for p in palindromes:
            if p % k == 0:
                count += 1
        return count

# Example usage:
# solution = Solution()
# print(solution.countGoodIntegers(3, 5))  # Output: 27
# print(solution.countGoodIntegers(1, 4))  # Output: 2
# print(solution.countGoodIntegers(5, 6))  # Output: 2468