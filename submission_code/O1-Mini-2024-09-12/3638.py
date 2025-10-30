class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter, defaultdict

        counts = Counter(s)
        max_f = max(counts.values())
        min_f = 1  # since f must be at least 1

        min_operations = float('inf')

        for f in range(min_f, max_f + 21):  # Adding a buffer to f
            carries = {0: 0}
            for i in range(26):
                char = chr(ord('a') + i)
                c = counts.get(char, 0)
                next_carries = {}
                for carry, ops in carries.items():
                    total = c + carry
                    if total > f:
                        surplus = total - f
                        # Option 1: Delete surplus
                        new_ops = ops + surplus
                        if 0 not in next_carries or new_ops < next_carries[0]:
                            next_carries[0] = new_ops
                        # Option 2: Change surplus to next character
                        new_ops = ops + surplus
                        if surplus not in next_carries or new_ops < next_carries[surplus]:
                            next_carries[surplus] = new_ops
                    elif total < f:
                        deficit = f - total
                        # Option: Insert deficit
                        new_ops = ops + deficit
                        if 0 not in next_carries or new_ops < next_carries[0]:
                            next_carries[0] = new_ops
                    else:
                        # No operation needed
                        if 0 not in next_carries or ops < next_carries[0]:
                            next_carries[0] = ops
                carries = next_carries
                if not carries:
                    break  # No possible way for this f

            # After all letters, any carry needs to be deleted
            for carry, ops in carries.items():
                total_ops = ops + carry
                if total_ops < min_operations:
                    min_operations = total_ops

        return min_operations