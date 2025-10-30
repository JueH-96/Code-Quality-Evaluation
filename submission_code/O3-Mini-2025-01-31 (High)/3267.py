class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        # Build contiguous segments for each character.
        segments = {}  # key: character, value: list of lengths of contiguous segments of that character.
        i = 0
        while i < n:
            ch = s[i]
            j = i
            while j < n and s[j] == ch:
                j += 1
            seg_length = j - i
            segments.setdefault(ch, []).append(seg_length)
            i = j

        ans = -1
        # For each letter, consider candidate special substring lengths.
        for ch in segments:
            seg_list = segments[ch]
            max_seg = max(seg_list)
            # Try each candidate length k from 1 to max_seg.
            for k in range(1, max_seg + 1):
                # Count occurrences of a k-length substring in each contiguous segment.
                # In a segment of length m, there are (m - k + 1) occurrences if m >= k.
                occurrences = sum(m - k + 1 for m in seg_list if m >= k)
                if occurrences >= 3:
                    ans = max(ans, k)
        return ans