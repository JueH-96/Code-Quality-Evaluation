class Solution:
    def finalString(self, s: str) -> str:
        screen = []
        for char in s:
            if char == 'i':
                screen.reverse()
            else:
                screen.append(char)
        return ''.join(screen)