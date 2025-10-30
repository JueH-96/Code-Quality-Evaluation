from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        # Map each substring to the set of indices of strings in which it appears
        subs_map = {}
        n = len(arr)
        
        for i, s in enumerate(arr):
            seen = set()
            L = len(s)
            for start in range(L):
                for end in range(start + 1, L + 1):
                    sub = s[start:end]
                    if sub in seen:
                        continue
                    seen.add(sub)
                    if sub not in subs_map:
                        subs_map[sub] = set()
                    subs_map[sub].add(i)
        
        answer = []
        # For each string, find its shortest unique substring
        for i, s in enumerate(arr):
            found = ""
            L = len(s)
            # Try lengths from 1 up to full length
            for length in range(1, L + 1):
                # Collect all substrings of this length
                candidates = []
                for start in range(L - length + 1):
                    sub = s[start:start + length]
                    # Check if it appears in any other string
                    if len(subs_map.get(sub, set())) == 1 and i in subs_map[sub]:
                        candidates.append(sub)
                if candidates:
                    # Pick the lexicographically smallest among the shortest
                    found = min(candidates)
                    break
            answer.append(found)
        
        return answer

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.shortestSubstrings(["cab","ad","bad","c"]))  # ["ab","","ba",""]
    print(sol.shortestSubstrings(["abc","bcd","abcd"]))    # ["","","abcd"]