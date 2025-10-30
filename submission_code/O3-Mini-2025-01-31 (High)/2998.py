class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s = str(num)
            # Only consider numbers with even number of digits.
            if len(s) % 2 != 0:
                continue
            
            n = len(s) // 2
            # Calculate the sum of the first half and second half of the digits.
            if sum(int(d) for d in s[:n]) == sum(int(d) for d in s[n:]):
                count += 1
                
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countSymmetricIntegers(1, 100))  # Expected output: 9
    print(sol.countSymmetricIntegers(1200, 1230))  # Expected output: 4