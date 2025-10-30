class Solution:
    def findValidPair(self, s: str) -> str:
        # Count frequency of each digit in the string
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        n = len(s)
        # Traverse each adjacent pair
        for i in range(n - 1):
            first, second = s[i], s[i+1]
            if first == second:
                continue  # Digits must be distinct in the pair.
            # Check if each digit appears in the string exactly as many times as its numeric value.
            if freq[first] == int(first) and freq[second] == int(second):
                return first + second
        return ""
        
# Example Test Cases
if __name__ == "__main__":
    sol = Solution()
    print(sol.findValidPair("2523533"))  # Expected Output: "23"
    print(sol.findValidPair("221"))      # Expected Output: "21"
    print(sol.findValidPair("22"))       # Expected Output: ""