class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 1000000007
        Len = len(nums) + 2

        sorted_pos = collections.defaultdict(list)
        for l, n in queries:
            sorted_pos[l].append(n)
        lengths = sorted(sorted_pos.keys())

        bit_left = [BinaryIndexedTree(Len) for _ in range(2)]
        bit_right = [BinaryIndexedTree(Len) for _ in range(2)]
        ans = 0

        for l in lengths:
            for n in sorted_pos[l][::-1]:
                a, b = -1, l
                while b - a > 1:
                    c = (a + b) // 2
                    if bit_left[1].query(c) >= bit_left[0].query(c) + n: b = c
                    else: a = c
                left_max = max(n, bit_left[0].query(l) + n, bit_left[1].query(b))

                a, b = l, Len
                while b - a > 1:
                    c = (a + b) // 2
                    if bit_right[1].query(c) >= bit_right[0].query(c) + n: a = c
                    else: b = c
                right_max = max(n, bit_right[0].query(l) + n, bit_right[1].query(a))

                bit_left[0].update(l + 1, right_max)
                bit_left[1].update(l + 1, left_max)
                bit_right[0].update(l + 1, left_max)
                bit_right[1].update(l + 1, right_max)

                ans += max(0, left_max + right_max - n)

        return ans % MOD