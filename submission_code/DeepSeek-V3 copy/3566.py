class Solution:
    def stringSequence(self, target: str) -> List[str]:
        if not target:
            return []
        
        result = []
        current = ""
        for char in target:
            if not current:
                current = "a"
                result.append(current)
                if current == char:
                    continue
                else:
                    # Need to increment 'a' to reach the target character
                    # Since we can only press key 2, which increments the last character
                    # We need to press key 2 multiple times
                    # For example, to get from 'a' to 'b', press key 2 once
                    # To get from 'a' to 'c', press key 2 twice, etc.
                    # So, the number of key presses is ord(char) - ord('a')
                    # But since each press changes the last character, we need to append the new character each time
                    # So, for each step, we append the new character to the result
                    # For example, to get from 'a' to 'c':
                    # 1. 'a' -> 'b' (press key 2 once)
                    # 2. 'b' -> 'c' (press key 2 once)
                    # So, the sequence is ['a', 'b', 'c']
                    # So, for each step, we append the new character to the result
                    # So, the number of steps is ord(char) - ord('a')
                    # So, we need to loop from ord('a') + 1 to ord(char)
                    for i in range(ord('a') + 1, ord(char) + 1):
                        current = current[:-1] + chr(i)
                        result.append(current)
            else:
                # Check if the current last character is the same as the target character
                if current[-1] == char:
                    # No need to press key 2, just press key 1 to append 'a'
                    current += 'a'
                    result.append(current)
                else:
                    # Need to press key 2 to change the last character to the target character
                    # Calculate the number of steps needed
                    # The last character is current[-1], and we need to reach char
                    # So, the number of steps is (ord(char) - ord(current[-1])) % 26
                    # Because 'z' wraps around to 'a'
                    steps = (ord(char) - ord(current[-1])) % 26
                    for _ in range(steps):
                        current = current[:-1] + chr((ord(current[-1]) - ord('a') + 1) % 26 + ord('a'))
                        result.append(current)
                    # Now, press key 1 to append 'a'
                    current += 'a'
                    result.append(current)
        # Remove the last 'a' if it's not part of the target
        # Because the last step is to press key 2 to change the last character to the target character
        # So, the last step is to press key 2, not key 1
        # So, the last 'a' is not part of the target
        # So, we need to remove it
        if result[-1][-1] == 'a' and result[-1][-1] != target[-1]:
            result.pop()
        return result