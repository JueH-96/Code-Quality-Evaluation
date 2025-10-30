class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            num_str = str(num)
            if len(num_str) % 2 == 0:
                n = len(num_str) // 2
                if sum(int(num_str[i]) for i in range(n)) == sum(int(num_str[i]) for i in range(n, 2 * n)):
                    count += 1
        return count