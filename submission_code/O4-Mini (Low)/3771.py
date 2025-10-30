class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        if k == 0:
            return True

        # 1. Compute first and last occurrences
        first = {}
        last = {}
        for i, ch in enumerate(s):
            if ch not in first:
                first[ch] = i
            last[ch] = i

        # 2. Find all minimal "valid" intervals
        intervals = []
        i = 0
        while i < n:
            ch = s[i]
            # try to start an interval only at the first occurrence of ch
            if first[ch] == i:
                end = last[ch]
                j = i
                # expand until no character inside forces us to extend
                while j <= end:
                    end = max(end, last[s[j]])
                    j += 1
                # check if it's not the entire string
                if not (i == 0 and end == n - 1):
                    intervals.append((i, end))
                # advance i to skip over this whole block
                i = j
            else:
                i += 1

        # 3. Greedily pick the maximum number of non‐overlapping intervals
        intervals.sort(key=lambda x: x[1])
        count = 0
        prev_end = -1
        for l, r in intervals:
            if l > prev_end:
                count += 1
                prev_end = r
                if count >= k:
                    return True

        return False


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxSubstringLength("abcdbaefab", 2))  # True
    print(sol.maxSubstringLength("cdefdc", 3))      # False
    print(sol.maxSubstringLength("abeabe", 0))      # True