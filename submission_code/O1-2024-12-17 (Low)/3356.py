class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)

        # Helper function to check if a substring occurs in any other string
        def occurs_in_others(sub: str, index: int) -> bool:
            for j in range(n):
                if j == index:
                    continue
                if sub in arr[j]:
                    return True
            return False

        answer = []
        for i in range(n):
            s = arr[i]
            found_substring = ""
            
            # We'll try substrings of increasing length
            # As soon as we find a valid one, we break.
            for length in range(1, len(s) + 1):
                # Collect all substrings of current length
                candidates = []
                for start in range(len(s) - length + 1):
                    candidates.append(s[start:start + length])
                
                # Sort candidates to ensure lexicographically smallest is checked first
                candidates = sorted(set(candidates))
                
                candidate_found = False
                for sub in candidates:
                    if not occurs_in_others(sub, i):
                        found_substring = sub
                        candidate_found = True
                        break
                if candidate_found:
                    break
            
            answer.append(found_substring)

        return answer