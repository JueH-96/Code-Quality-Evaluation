class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        # run[i] = length of the maximal run of the same character starting at i
        run = [0] * n
        run[-1] = 1
        for i in range(n-2, -1, -1):
            if s[i] == s[i+1]:
                run[i] = run[i+1] + 1
            else:
                run[i] = 1

        ans = 0
        # Only characters that actually appear in s
        for c in set(s):
            # collect run lengths at positions where s[i] == c
            runs_c = [run[i] for i in range(n) if s[i] == c]
            max_run = max(runs_c)
            # Try possible lengths from largest down to 1
            for k in range(max_run, 0, -1):
                # count how many positions can support a run of length k
                cnt = sum(1 for r in runs_c if r >= k)
                if cnt >= 3:
                    ans = max(ans, k)
                    break  # no need to try smaller k for this character

        return ans if ans > 0 else -1