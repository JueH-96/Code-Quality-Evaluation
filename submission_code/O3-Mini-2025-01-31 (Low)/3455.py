class Solution:
    def minimumLength(self, s: str) -> int:
        # Observation:
        # In one operation you choose an index i whose character appears on both left and right.
        # Let c = s[i]. By the rule, you remove the closest occurrence of c to the left of i
        # and the closest occurrence of c to the right of i.
        # If you “look at” the string s and focus on all the positions where a fixed letter c appears,
        # notice that every operation involving c removes two occurrences that are consecutive in the 
        # sorted order of positions where c appears. (The chosen “pivot” remains.) 
        # Since the rule forces you to remove only from the “inside” (i.e. you must have one occurrence 
        # before and one after the chosen pivot), you cannot remove the first or last occurrence of c.
        #
        # Hence, for every letter, if it appears less than 3 times, no operation can be performed on it,
        # so all occurrences must remain.
        # If it appears at least 3 times then we can “remove” occurrences in pairs until only either
        # one or two remain. (You can think of it as “peeling off” the outer occurrences.)
        # By working through small examples one finds that:
        #   For k occurrences:
        #      if k == 1 or k == 2: final count = k   (since no move is possible in k==2)
        #      if k >= 3:
        #           if k is odd, then you can remove in pairs leaving exactly 1 occurrence.
        #           if k is even, you end up with 2 occurrences.
        #
        # With that in mind, the minimum final length is the sum over every character c of f(k)
        # where k is the number of occurrences of c and f is defined as:
        #      f(k) = k,            if k < 3
        #             1,            if k >= 3 and k is odd
        #             2,            if k >= 3 and k is even
        #
        # This answer is computed in O(n) time and O(1) auxiliary space (since there are 26 letters).
        
        counts = {}
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
        res = 0
        for cnt in counts.values():
            if cnt < 3:
                res += cnt
            else:
                if cnt % 2 == 1:
                    res += 1
                else:
                    res += 2
        return res
         
# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Provided test cases:
    print(sol.minimumLength("abaacbcbb"))  # Expected output: 5
    print(sol.minimumLength("aa"))         # Expected output: 2