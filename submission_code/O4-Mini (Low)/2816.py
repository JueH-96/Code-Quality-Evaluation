class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert string to list for easy modification
        arr = list(s)
        n = len(arr)
        
        # Only need to iterate over the first half
        for i in range(n // 2):
            j = n - 1 - i
            if arr[i] != arr[j]:
                # Choose the lexicographically smaller of the two characters
                smaller = min(arr[i], arr[j])
                arr[i] = smaller
                arr[j] = smaller
        
        # Join back into a string and return
        return "".join(arr)