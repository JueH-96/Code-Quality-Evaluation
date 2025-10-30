from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        result = []

        # Precompute all sets of strings for membership checks (excluding self)
        # Actually, to confirm uniqueness, we want to check in any other string, so
        # we don't necessarily need more advanced indexing, just "in" checks might suffice
        # given input size constraints.

        for i in range(n):
            s = arr[i]
            found = ""
            # We will look for the shortest substring by increasing length
            # and for each length, consider them in lexicographical order.
            substr_found = False
            for length in range(1, len(s) + 1):
                # get all substrings of length `length`
                subs = []
                for start in range(len(s) - length + 1):
                    subs.append(s[start:start+length])
                # sort them lexicographically
                subs = sorted(set(subs))
                
                # check for each substring if it appears in any other string
                for sub in subs:
                    exists_elsewhere = False
                    for j in range(n):
                        if j != i and sub in arr[j]:
                            exists_elsewhere = True
                            break
                    if not exists_elsewhere:
                        found = sub
                        substr_found = True
                        break  # Found the needed substring; stop.
                        
                if substr_found:
                    break
            
            result.append(found)
        
        return result