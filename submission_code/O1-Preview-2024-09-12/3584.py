class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        N = len(word1)
        M = len(word2)
        next_char = [ [ N ] * 26 for _ in range(N + 1) ]   # N+1 to avoid index error

        # Initialize the next occurrence of each character to N (end of string)
        for c in range(26):
            next_char[N][c] = N

        # Build next_char[i][c], where c is 0..25
        for i in range(N -1, -1, -1):
            # Copy the next_char from the position i+1
            for c in range(26):
                next_char[i][c] = next_char[i+1][c]
            # Update the next occurrence of the current character
            next_char[i][ ord(word1[i]) - ord('a') ] = i

        indices = []
        p = 0
        mismatches_used = 0

        for i in range(M):
            c = ord(word2[i]) - ord('a')
            if next_char[p][c] == p:  # word1[p] == word2[i]
                indices.append(p)
                p += 1
            else:
                if mismatches_used == 0:
                    # Use current position p as a mismatch
                    if p >= N:
                        return []
                    indices.append(p)
                    mismatches_used +=1
                    p += 1
                else:
                    # Try to find the next occurrence of the required character
                    if next_char[p][c] < N:
                        indices.append(next_char[p][c])
                        p = next_char[p][c] +1
                    else:
                        return []
        return indices