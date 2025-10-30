class Solution:
    def maximumLength(self, s: str) -> int:
        max_len = -1
        for char_code in range(ord('a'), ord('z') + 1):
            char = chr(char_code)
            run_lengths = []
            current_run_length = 0
            for c in s:
                if c == char:
                    current_run_length += 1
                else:
                    if current_run_length > 0:
                        run_lengths.append(current_run_length)
                    current_run_length = 0
            if current_run_length > 0:
                run_lengths.append(current_run_length)

            for k in range(1, 51):
                occurrences = 0
                for run_length in run_lengths:
                    if run_length >= k:
                        occurrences += (run_length - k + 1)
                if occurrences >= 3:
                    max_len = max(max_len, k)
        return max_len