class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # Helper function to compute the cyclic distance between two characters.
        def cyclic_distance(a: str, b: str) -> int:
            diff = abs(ord(a) - ord(b))
            return min(diff, 26 - diff)
        
        n = len(s)
        ans = []
        # Process each character in the input string.
        for i in range(n):
            cur = s[i]
            # Try each candidate letter from 'a' to 'z'
            for candidate in "abcdefghijklmnopqrstuvwxyz":
                cost = cyclic_distance(cur, candidate)
                # If we can afford this cost, pick candidate.
                if cost <= k:
                    ans.append(candidate)
                    k -= cost
                    # Once chosen, break out of candidate loop.
                    break
        return "".join(ans)


# The following is provided for local testing using standard input.
if __name__ == "__main__":
    import sys
    data = sys.stdin.read().split()
    if len(data) >= 2:
        s = data[0]
        k = int(data[1])
        sol = Solution()
        print(sol.getSmallestString(s, k))