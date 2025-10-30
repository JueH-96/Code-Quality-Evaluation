class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern p into two parts: prefix and suffix based on '*'
        pre, suf = p.split('*')
        n = len(s)
        # Check every possible non-empty substring of s
        for i in range(n):
            for j in range(i + 1, n + 1):
                candidate = s[i:j]
                # The candidate must be at least as long as the fixed parts of p,
                # because star can only expand (or remain empty) but cannot shrink.
                if len(candidate) < len(pre) + len(suf):
                    continue
                # Check if candidate starts with the prefix (pre) and ends with the suffix (suf)
                if candidate.startswith(pre) and candidate.endswith(suf):
                    return True
        return False

# Example testing through a driver code:
def solve():
    import sys
    data = sys.stdin.read().split()
    if len(data) < 2:
        return
    s, p = data[0], data[1]
    sol = Solution()
    print("true" if sol.hasMatch(s, p) else "false")

if __name__ == '__main__':
    solve()