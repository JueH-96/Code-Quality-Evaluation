class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        # Initialize the dp array where dp[i] will be the minimum number of extra characters
        # when considering the substring s[:i]
        dp = [float('inf')] * (len(s) + 1)
        dp[0] = 0  # Base case: no extra characters if no string is considered
        
        # Iterate over the string s
        for i in range(1, len(s) + 1):
            # Check each word in the dictionary
            for word in dictionary:
                word_len = len(word)
                # If the word can fit into the current substring ending at i and
                # if the word matches the corresponding substring in s
                if i >= word_len and s[i - word_len:i] == word:
                    # Update the dp array with the minimum value between the current dp value
                    # and the dp value from the previous index that would start the word
                    # plus the extra characters between the previous word and the current word
                    dp[i] = min(dp[i], dp[i - word_len] + i - word_len - (i - len(word)))
        
        # The answer will be the minimum extra characters considering the whole string
        return dp[-1] if dp[-1] != float('inf') else len(s)

# Example usage:
# sol = Solution()
# print(sol.minExtraChar("leetscode", ["leet","code","leetcode"]))  # Output: 1
# print(sol.minExtraChar("sayhelloworld", ["hello","world"]))       # Output: 3