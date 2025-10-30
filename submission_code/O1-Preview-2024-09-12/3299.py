class Solution:
    def maximumLength(self, nums):
        import math
        from collections import Counter, defaultdict

        counts = Counter(nums)
        max_length = 1  # At least we can choose one number

        MAX_NUM = 10 ** 9

        for x in counts.keys():
            exponents = []
            y = x
            while y <= MAX_NUM:
                exponents.append(y)
                y_next = y * y
                if y_next == y:  # Prevent infinite loop if y is 1
                    break
                y = y_next

            t_max = len(exponents) - 1

            for t in range(t_max + 1):
                counts_needed = defaultdict(int)
                for i in range(t + 1):
                    y_val = exponents[i]
                    if i == t:
                        counts_needed[y_val] += 1  # Center element
                    else:
                        counts_needed[y_val] += 2
                success = True
                for y_val in counts_needed:
                    if counts[y_val] >= counts_needed[y_val]:
                        continue
                    else:
                        success = False
                        break
                if success:
                    length_candidate = 2 * t + 1
                    if length_candidate > max_length:
                        max_length = length_candidate
                else:
                    break

        return max_length