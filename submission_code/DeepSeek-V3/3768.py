class Solution:
    def hasSameDigits(self, s: str) -> bool:
        current = s
        while len(current) > 2:
            next_s = []
            for i in range(len(current) - 1):
                digit1 = int(current[i])
                digit2 = int(current[i+1])
                new_digit = (digit1 + digit2) % 10
                next_s.append(str(new_digit))
            current = ''.join(next_s)
        return current[0] == current[1]