class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # special case: if k == 1, every substring is valid.
        if k == 1:
            return n * (n + 1) // 2
        
        total = 0
        
        for i in range(n):
            freq = [0] * 26
            valid_found = False
            # iterate j from i to end; once valid is found, all substrings from i with index >= j are valid
            for j in range(i, n):
                idx = ord(s[j]) - ord('a')
                freq[idx] += 1
                # check if this update made the substring valid (i.e. any frequency reached k)
                if freq[idx] == k:
                    # substring s[i:j+1] is the first valid one from left index i
                    total += (n - j)
                    break  # break out of j loop, move to next starting index i
                
        return total