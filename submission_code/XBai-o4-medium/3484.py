class Solution:
    def getSmallestString(self, s: str) -> str:
        candidates = [s]
        for i in range(len(s) - 1):
            # Check if current and next character have the same parity
            current = int(s[i])
            next_val = int(s[i+1])
            if (current % 2) == (next_val % 2):
                # Create a new swapped string
                s_list = list(s)
                s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                new_s = ''.join(s_list)
                candidates.append(new_s)
        # Return the minimum candidate lexicographically
        return min(candidates)