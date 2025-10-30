class Solution:
    def minimumChairs(self, s: str) -> int:
        current = 0
        maximum = 0
        for ch in s:
            if ch == 'E':
                current += 1
                if current > maximum:
                    maximum = current
            else:  # ch == 'L'
                current -= 1
        return maximum

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumChairs("EEEEEEE"))   # Output: 7
    print(sol.minimumChairs("ELELEEL"))   # Output: 2
    print(sol.minimumChairs("ELEELEELLL"))# Output: 3