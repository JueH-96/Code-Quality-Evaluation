class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # The idea is to greedily replace each character of s with the lexicographically smallest
        # candidate letter from 'a' to 'z' that we can transform into without exceeding the remaining budget.
        # For a given character s[i] and a candidate letter cand, the cost is defined as:
        #    cost = min(|ord(s[i]) - ord(cand)|, 26 - |ord(s[i]) - ord(cand)|)
        # Since we can always choose to leave later letters unchanged (cost = 0),
        # a candidate is feasible as long as its individual cost does not exceed the remaining budget.
        # By checking candidates from 'a' upward, we ensure that each letter of t is as small as possible.
        
        res = []
        rem = k  # remaining budget
        
        for ch in s:
            # Try every candidate from 'a' to 'z'
            for candidate in (chr(c) for c in range(ord('a'), ord('z') + 1)):
                diff = abs(ord(candidate) - ord(ch))
                # Compute circular cost
                cost = min(diff, 26 - diff)
                if cost <= rem:
                    res.append(candidate)
                    rem -= cost
                    break  # move on to next letter in s
        return "".join(res)