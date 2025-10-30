class Solution:
    def minimizedStringLength(self, s: str) -> int:
        chars = list(s)

        while True:
            operated = False
            n = len(chars)
            indices_to_remove = set()

            for i in range(n):
                if i >= len(chars):  # Check if index is still valid
                    break

                c = chars[i]

                left_idx = None
                for j in range(i - 1, -1, -1):
                    if j < len(chars) and chars[j] == c:
                        left_idx = j
                        break

                right_idx = None
                for j in range(i + 1, len(chars)):
                    if j < len(chars) and chars[j] == c:
                        right_idx = j
                        break

                if left_idx is not None and right_idx is not None:
                    if left_idx not in indices_to_remove and right_idx not in indices_to_remove:
                        indices_to_remove.add(left_idx)
                        indices_to_remove.add(right_idx)
                        operated = True
                        break  # Perform one operation per iteration

            if not operated:
                break

            new_chars = [char for idx, char in enumerate(chars) if idx not in sorted(list(indices_to_remove))]
            chars = new_chars

        return len(chars)