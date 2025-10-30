class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        if n == 2:
            # If there are only two levels, Alice can only win if she plays a possible level and Bob plays an impossible one
            if possible[0] == 1 and possible[1] == 0:
                return 1
            else:
                return -1
        
        # Calculate the score changes for each level
        score_changes = [1 if x == 1 else -1 for x in possible]
        
        # Calculate prefix sums of score changes
        prefix_sums = [0] * n
        prefix_sums[0] = score_changes[0]
        for i in range(1, n):
            prefix_sums[i] = prefix_sums[i-1] + score_changes[i]
        
        # Calculate suffix sums of score changes
        suffix_sums = [0] * n
        suffix_sums[-1] = score_changes[-1]
        for i in range(n-2, -1, -1):
            suffix_sums[i] = suffix_sums[i+1] + score_changes[i]
        
        # Alice plays from 0 to i, Bob plays from i+1 to n-1
        # Alice's score: prefix_sums[i]
        # Bob's score: suffix_sums[i+1] if i+1 <= n-1 else 0
        for i in range(n-1):
            alice_score = prefix_sums[i]
            bob_score = suffix_sums[i+1] if i+1 < n else 0
            if alice_score > bob_score:
                return i + 1
        
        return -1