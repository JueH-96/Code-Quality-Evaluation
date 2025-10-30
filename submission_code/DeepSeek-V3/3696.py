class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0
        n = len(s)
        for i in range(n):
            last_char = s[i]
            d = int(last_char)
            if d == 0:
                continue  # division by zero is not allowed
            current_num = 0
            power = 1  # 10^0 initially, but we process digits from i backwards to j
            # We process substrings s[j..i] for j from i down to 0
            for j in range(i, -1, -1):
                digit = int(s[j])
                current_num += digit * power
                power *= 10
                if current_num % d == 0:
                    count += 1
                # To prevent power and current_num from getting too large, we can take mod d periodically
                # Since (a + b) mod m = (a mod m + b mod m) mod m, and (a * b) mod m = ((a mod m) * (b mod m)) mod m
                current_num %= d
                power %= d
        return count