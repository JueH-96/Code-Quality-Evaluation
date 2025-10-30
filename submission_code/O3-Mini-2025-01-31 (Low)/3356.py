from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        # Pre-calculate all substrings for each word.
        substrings_list = []
        for word in arr:
            subs = set()  # using set to avoid duplicates
            L = len(word)
            for i in range(L):
                for j in range(i+1, L+1):
                    subs.add(word[i:j])
            # sort by length then lexicographically
            sorted_subs = sorted(list(subs), key=lambda s: (len(s), s))
            substrings_list.append(sorted_subs)
        
        # Prepare the answer list.
        ans = ["" for _ in range(n)]
        
        # For each string, test each candidate substring.
        for i in range(n):
            word = arr[i]
            found = False
            for candidate in substrings_list[i]:
                unique = True
                for j in range(n):
                    # skip self-comparison
                    if i == j:
                        continue
                    # if candidate is in the other word, mark not unique
                    if candidate in arr[j]:
                        unique = False
                        break
                if unique:
                    ans[i] = candidate
                    found = True
                    break
            if not found:
                ans[i] = ""  # empty string if no candidate found
        return ans

# For running quick tests:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    arr1 = ["cab","ad","bad","c"]
    print(sol.shortestSubstrings(arr1))  # Expected: ["ab", "", "ba", ""]
    
    # Example 2
    arr2 = ["abc","bcd","abcd"]
    print(sol.shortestSubstrings(arr2))  # Expected: ["", "", "abcd"]