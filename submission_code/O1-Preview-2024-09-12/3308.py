class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        positions = [[] for _ in range(26)]
        n = len(s)
        for idx, ch in enumerate(s):
            positions[ord(ch) - ord('a')].append(idx)
        
        counters = [0]*26  # pointers to next occurrence to remove
        positions_not_removed = set(range(n))
        
        while True:
            positions_to_remove = []
            for c in range(26):
                if counters[c] < len(positions[c]):
                    positions_to_remove.append(positions[c][counters[c]])
            if len(positions_to_remove) == 0:
                # Nothing to remove, positions_not_removed is current s
                return ''.join(s[idx] for idx in sorted(positions_not_removed))
            if len(positions_not_removed) <= len(positions_to_remove):
                # After this operation, s will become empty
                # So we need to output current s
                return ''.join(s[idx] for idx in sorted(positions_not_removed))
            # Remove positions
            for pos in positions_to_remove:
                positions_not_removed.remove(pos)
            # Update counters
            for c in range(26):
                if counters[c] < len(positions[c]):
                    counters[c] += 1