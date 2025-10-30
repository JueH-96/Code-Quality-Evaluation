class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        arr = list(s)
        n = len(arr)
        for i in range(n // 2):
            j = n - i - 1
            if arr[i] != arr[j]:
                # Replace both characters with the lexicographically smaller one
                chosen = arr[i] if arr[i] < arr[j] else arr[j]
                arr[i] = chosen
                arr[j] = chosen
        return "".join(arr)

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.makeSmallestPalindrome("egcfe"))  # Output: "efcfe"
    print(sol.makeSmallestPalindrome("abcd"))   # Output: "abba"
    print(sol.makeSmallestPalindrome("seven"))  # Output: "neven"