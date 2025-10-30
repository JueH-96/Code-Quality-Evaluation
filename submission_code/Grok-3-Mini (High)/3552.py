class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        m = (n + 1) // 2
        high = 10 ** m - 1
        low = 10 ** (m - 1)
        for num in range(high, low - 1, -1):
            H_str = str(num)
            # Compute P mod k
            val = 0
            for i in range(n):
                idx = min(i, n - 1 - i)
                digit = int(H_str[idx])
                val = (val * 10 + digit) % k
            if val == 0:
                # Build P_str
                P_str_list = []
                for i in range(n):
                    idx_min = min(i, n - 1 - i)
                    digit_char = H_str[idx_min]
                    P_str_list.append(digit_char)
                return ''.join(P_str_list)