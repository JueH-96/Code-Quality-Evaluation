class Solution:
    def calculateScore(self, s: str) -> int:
        mirror_map = {chr(97 + i): chr(122 - i) for i in range(26)}
        reverse_map = {y: x for x, y in mirror_map.items()}
        mirror_map.update(reverse_map)

        stack = []
        score = 0

        for i, char in enumerate(s):
            if char in mirror_map:
                found = False
                while stack and not found:
                    j, mirrored_char = stack.pop()
                    if mirrored_char == mirror_map[char]:
                        score += i - j
                        found = True
                    else:
                        stack.append((j, mirrored_char))
                if not found:
                    stack.append((i, char))

        return score