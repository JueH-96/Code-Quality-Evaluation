class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)

        # Check if the parity of the number of differing bits is the same
        diff_count = sum(1 for i in range(n) if s1[i] != s2[i])
        if diff_count % 2 != 0:
            return -1

        # Initialize cost
        cost = 0

        # Perform operations to make s1 equal to s2
        i = 0
        while i < n:
            if s1[i] != s2[i]:
                # Find the next differing bit
                j = i + 1
                while j < n and s1[j] == s2[j]:
                    j += 1

                if j < n:
                    # Flip s1[i] and s1[j]
                    cost += min(x, (j - i) * 2)
                    s1 = s1[:i] + s2[i] + s1[i+1:j] + s2[j] + s1[j+1:]
                    i = j
                else:
                    # If no more differing bits, flip the remaining
                    cost += x
                    s1 = s1[:i] + s2[i] + s1[i+1:]

            i += 1

        return cost