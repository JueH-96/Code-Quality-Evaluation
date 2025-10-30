class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        freqs = sorted(count.values())
        n = len(freqs)
        ans = float('inf')
        for num_groups in range(1, freqs[0] + 1):
            valid = True
            total_groups = 0
            for freq in freqs:
                if freq < num_groups:
                    valid = False
                    break
                q, r = divmod(freq, num_groups + 1)
                if r > q:
                    valid = False
                    break
                total_groups += q + (r > 0)

            if valid:
                ans = min(ans, total_groups)
        return ans