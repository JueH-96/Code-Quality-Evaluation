class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        # Convert the levels into +1/-1 based on whether each level is passable or not.
        # +1 if possible[i] == 1, -1 otherwise.
        arr = [1 if x == 1 else -1 for x in possible]
        n = len(arr)
        
        # Compute the prefix sums, prefix[i] = sum of arr[:i].
        prefix = [0] * (n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + arr[i]
        
        total_sum = prefix[n]
        
        # We must ensure Bob plays at least 1 level, so Alice can play from 1 to n-1 levels
        # i.e., the prefix index i ranges from 1..n-1 in prefix array
        # Condition for Alice to have more points than Bob:
        #   prefix[i] > (total_sum - prefix[i])   =>  2 * prefix[i] > total_sum
        for i in range(1, n):  # i goes up to n-1 (so that Bob has at least 1 level to play)
            if 2 * prefix[i] > total_sum:
                return i  # i is the number of levels Alice plays
        
        return -1