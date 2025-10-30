class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # The key observation is that in each operation, for every letter, we remove its first (or
        # “next”) occurrence. That means for each letter, its occurrences are removed in order:
        # the 1st occurrence is removed in round 1, the 2nd in round 2, and so on.
        #
        # If we go through the string and assign a "removal round" to each character—its order of 
        # appearance among the same letter—then the overall number of rounds needed is the maximum 
        # removal round across all characters. In the final operation (say round R), the string that 
        # we started with will consist exactly of those characters that are removed in the R-th round.
        #
        # Thus, our task reduces to:
        #   1. Iterate over the string and for each character track how many times it has shown up 
        #      so far. This number is its removal round.
        #   2. Find the maximum removal round (R).
        #   3. Build a result string from all characters whose removal round equals R (in original order).
        
        count = {}
        removal_round = [0] * len(s)
        
        # Assign removal rounds: for each letter, its j-th occurrence gets removed in round j.
        for i, ch in enumerate(s):
            cnt = count.get(ch, 0) + 1
            removal_round[i] = cnt
            count[ch] = cnt
        
        # The final round is the maximum number among all removal rounds.
        max_round = max(removal_round)
        
        # Build the answer string with all characters that are removed in the final round.
        result = []
        for i, ch in enumerate(s):
            if removal_round[i] == max_round:
                result.append(ch)
        return "".join(result)


# The following part is for testing and I/O processing.
def solve():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0].strip()
    
    # Using our defined method from the Solution class.
    sol = Solution()
    result = sol.lastNonEmptyString(s)
    sys.stdout.write(result)
    
if __name__ == '__main__':
    solve()