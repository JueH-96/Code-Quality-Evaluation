class Solution:
    def minimumSteps(self, s: str) -> int:
        black_count = s.count('1')
        white_count = s.count('0')
        black_prefix_count = 0
        white_prefix_count = 0
        min_steps = float('inf')

        for i in range(len(s)):
            if s[i] == '1':
                black_prefix_count += 1
            else:
                white_prefix_count += 1

            steps = min(black_prefix_count + white_count - white_prefix_count,
                        white_prefix_count + black_count - black_prefix_count)
            min_steps = min(min_steps, steps)

        return min_steps