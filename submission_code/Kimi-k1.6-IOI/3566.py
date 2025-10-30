class Solution:
    def stringSequence(self, target: str) -> List[str]:
        current = []
        result = []
        for c in target:
            # Press Key1 to append 'a'
            current.append('a')
            result.append(''.join(current))
            # Calculate the number of Key2 presses needed
            diff = (ord(c) - ord('a')) % 26
            for _ in range(diff):
                # Increment the last character using Key2
                last_char = current[-1]
                new_char = chr((ord(last_char) - ord('a') + 1) % 26 + ord('a'))
                current[-1] = new_char
                result.append(''.join(current))
        return result