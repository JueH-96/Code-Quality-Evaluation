class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        # Function to group the string into consecutive identical characters
        def group_lengths(s):
            groups = []
            n = len(s)
            if n == 0:
                return groups
            current = s[0]
            count = 1
            for i in range(1, n):
                if s[i] == current:
                    count += 1
                else:
                    groups.append(count)
                    current = s[i]
                    count = 1
            groups.append(count)
            return groups
        
        groups = group_lengths(s)
        n = len(s)
        
        # Binary search on the possible L
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            flips_needed = 0
            for g in groups:
                if g > mid:
                    flips_needed += (g + mid - 1) // mid - 1
            if flips_needed <= numOps:
                right = mid
            else:
                left = mid + 1
        return left