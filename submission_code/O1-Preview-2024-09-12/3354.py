class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        counts = [0]*26  # counts for 'a' to 'z'
        s_list = list(s)

        for i in range(n):
            if s_list[i] != '?':
                counts[ord(s_list[i]) - ord('a')] +=1
            else:
                # Try to find a letter that hasn't appeared yet
                found = False
                for idx in range(26):
                    if counts[idx]==0:
                        s_list[i] = chr(ord('a') + idx)
                        counts[idx] +=1
                        found = True
                        break
                if not found:
                    # All letters have appeared, pick the one with minimal count
                    min_count = min(counts)
                    candidates = [idx for idx, cnt in enumerate(counts) if cnt == min_count]
                    s_list[i] = chr(ord('a') + candidates[0])  # Pick the lex smallest
                    counts[candidates[0]] +=1

        return ''.join(s_list)