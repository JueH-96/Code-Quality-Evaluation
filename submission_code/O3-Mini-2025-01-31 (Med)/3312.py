class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        
        count = 0
        # Start from the first character and compare with previous character ignoring case.
        prev = s[0].lower()
        for char in s[1:]:
            current = char.lower()
            if current != prev:
                count += 1
            prev = current
        
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countKeyChanges("aAbBcC"))  # Output: 2
    print(sol.countKeyChanges("AaAaAaaA"))  # Output: 0