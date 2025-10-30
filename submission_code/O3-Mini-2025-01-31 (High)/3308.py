from collections import Counter

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # In each “round” of removals, for every letter that is present,
        # we remove its first (i.e. left‐most) occurrence from the current string.
        # Notice that for each letter c that occurs in s, its removals follow
        # the order of its occurrences. In the 1st round we remove its 1st occurrence,
        # in the 2nd round its 2nd occurrence, and so on.
        # Thus, if we let f(c) be the total frequency of c in s,
        # then after f(c) rounds, letter c is completely removed.
        #
        # The whole string s becomes empty after r rounds, where 
        # r = max{ f(c) for each letter c in s }.
        # We are asked for the state of s right before the last (r-th) round.
        #
        # Look at the fate of each letter c:
        # • If f(c) < r then c is completely removed before round r.
        # • If f(c) == r then after r-1 rounds, exactly one copy of c remains – 
        #   namely, its r-th occurrence in the original s.
        #
        # And the removals are done simultaneously preserving the 
        # relative original order of the survivors.
        # Therefore, the answer is exactly the sequence of the (r-th) occurrences
        # (i.e. the last occurrence) for every letter c with f(c)==r, arranged in 
        # the order in which they appear in s.
        
        freq = Counter(s)
        r = max(freq.values())
        
        # For every letter that appears exactly r times (the maximum),
        # its r-th occurrence in s is its last appearance.
        # We simply record the index of the last occurrence (using s.rfind)
        # and then sort these letters by that index.
        survivors = []
        for c in freq:
            if freq[c] == r:
                pos = s.rfind(c)
                survivors.append((pos, c))
        
        survivors.sort(key=lambda x: x[0])
        return "".join(letter for _, letter in survivors)
        
# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    s1 = "aabcbbca"
    print(sol.lastNonEmptyString(s1))  # Expected output: "ba"
    
    # Example 2
    s2 = "abcd"
    print(sol.lastNonEmptyString(s2))  # Expected output: "abcd"