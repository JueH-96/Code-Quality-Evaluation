class Solution:
    def possibleStringCount(self, word: str) -> int:
        # We want to count the number of distinct intended strings.
        # Alice may have long-pressed one key group "at most once".
        # That means: for every contiguous group of the same character (with count k):
        # - If no long press error occurs for that group, then the intended group is exactly k characters.
        # - If a long press error occurred in that group (allowed only for at most one group overall), 
        #   then she intended to type strictly fewer than k characters, but at least 1.
        # So for a group with count k > 1, there are (k - 1) possibilities if it was the error group.
        # For a group with k == 1, there is no possibility of an error because pressing for too long
        # would result in more than 1 character.
        # Finally we add the possibility that no error occurred at all (i.e. all groups are exactly as typed).
        # This yields the formula:
        # answer = 1 (no error anywhere) + sum_{group with count > 1} (count - 1)
        
        n = len(word)
        if n == 0:
            return 0
        
        groups = []
        count = 1
        for i in range(1, n):
            if word[i] == word[i - 1]:
                count += 1
            else:
                groups.append(count)
                count = 1
        groups.append(count)  # append the last group's count
        
        result = 1  # case of no error anywhere
        for group_count in groups:
            if group_count > 1:
                result += (group_count - 1)
        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.possibleStringCount("abbcccc"))  # Output: 5
    print(sol.possibleStringCount("abcd"))     # Output: 1
    print(sol.possibleStringCount("aaaa"))     # Output: 4