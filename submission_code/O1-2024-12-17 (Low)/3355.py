class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Each player must play at least 1 level, so n >= 2
        
        # Convert the array into +1 (if possible[i] == 1) or -1 (if possible[i] == 0)
        arr = [1 if x == 1 else -1 for x in possible]

        # Compute prefix sums: pref[i] = arr[0] + arr[1] + ... + arr[i]
        pref = [0] * n
        pref[0] = arr[0]
        for i in range(1, n):
            pref[i] = pref[i-1] + arr[i]

        # Compute suffix sums: suff[i] = arr[i] + arr[i+1] + ... + arr[n-1]
        suff = [0] * n
        suff[n-1] = arr[n-1]
        for i in range(n-2, -1, -1):
            suff[i] = suff[i+1] + arr[i]

        # We need to find the smallest index i (0 <= i <= n-2) such that
        # pref[i] > suff[i+1].  The number of levels Alice plays is i+1.
        for i in range(n-1):
            # Bob must get at least 1 level => i cannot be n-1
            if i < n-1:  
                if pref[i] > suff[i+1]:
                    return i + 1

        return -1