class Solution:
    def getSmallestString(self, s: str) -> str:
        candidates = []
        n = len(s)
        for i in range(n - 1):
            a = s[i]
            b = s[i + 1]
            if (int(a) % 2) == (int(b) % 2):
                # Create a new string by swapping i and i+1
                new_list = list(s)
                new_list[i], new_list[i + 1] = new_list[i + 1], new_list[i]
                new_s = ''.join(new_list)
                candidates.append(new_s)
        # If no swaps possible, return original
        if not candidates:
            return s
        # Include original in candidates to find the minimal
        return min(candidates + [s])