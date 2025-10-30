class Solution:
    def punishmentNumber(self, n: int) -> int:
        total = 0
        for i in range(1, n + 1):
            square = i * i
            s = str(square)
            m = len(s)
            valid = False
            # Iterate over all possible split masks
            for mask in range(0, 1 << (m - 1)):
                parts = []
                current = []
                for j in range(m):
                    current.append(s[j])
                    if j < m - 1 and (mask & (1 << j)):
                        parts.append(''.join(current))
                        current = []
                if current:
                    parts.append(''.join(current))
                # Calculate the sum of the parts
                sum_parts = sum(int(part) for part in parts)
                if sum_parts == i:
                    valid = True
                    break
            if valid:
                total += square
        return total