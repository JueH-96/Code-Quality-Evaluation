class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current_str = ""
        for i in range(len(target)):
            current_char = target[i]
            # Append 'a' using key 1
            current_str += 'a'
            result.append(current_str)
            # Calculate steps needed to reach current_char from 'a'
            steps_needed = ord(current_char) - ord('a')
            for _ in range(steps_needed):
                # Update the last character using key 2
                current_str = current_str[:-1] + chr(ord(current_str[-1]) + 1)
                result.append(current_str)
        return result