class Solution:
    def minLength(self, s: str) -> int:
        s_list = list(s)
        removed = 0
        while True:
            found = False
            for i in range(len(s_list) - 1):
                if (s_list[i] == 'A' and s_list[i+1] == 'B') or (s_list[i] == 'C' and s_list[i+1] == 'D'):
                    # Remove the two characters
                    s_list.pop(i+1)
                    s_list.pop(i)
                    removed += 1
                    found = True
                    break
            if not found:
                break
        return len(s) - 2 * removed