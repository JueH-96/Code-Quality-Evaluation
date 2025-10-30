class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Continue reducing until only 2 digits remain
        while len(s) > 2:
            # Compute the next string by summing consecutive digit pairs modulo 10
            next_digits = []
            for i in range(len(s) - 1):
                summed = (int(s[i]) + int(s[i+1])) % 10
                next_digits.append(str(summed))
            s = "".join(next_digits)
        
        # Check if the final two digits are the same
        return s[0] == s[1]


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.hasSameDigits("3902"))  # True
    print(sol.hasSameDigits("34789")) # False