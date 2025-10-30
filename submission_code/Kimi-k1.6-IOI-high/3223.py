class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        # Split the word into segments where adjacent characters differ by at most 2
        segments = []
        n = len(word)
        if n == 0:
            return 0
        current_segment = [word[0]]
        for i in range(1, n):
            prev_char = word[i-1]
            curr_char = word[i]
            if abs(ord(curr_char) - ord(prev_char)) <= 2:
                current_segment.append(curr_char)
            else:
                segments.append(''.join(current_segment))
                current_segment = [curr_char]
        segments.append(''.join(current_segment))
        
        total = 0
        
        # Process each segment
        for seg in segments:
            seg_len = len(seg)
            if seg_len == 0:
                continue
            
            # Precompute prefix sums for each character in the segment
            prefix = [[0] * (seg_len + 1) for _ in range(26)]
            for i in range(seg_len):
                c = ord(seg[i]) - ord('a')
                for j in range(26):
                    prefix[j][i+1] = prefix[j][i] + (1 if j == c else 0)
            
            # Check all possible m values (1 to 26)
            for m in range(1, 27):
                required_length = m * k
                if required_length > seg_len:
                    continue
                # Calculate the number of possible starting indices
                max_start = seg_len - required_length
                if max_start < 0:
                    continue
                for start in range(max_start + 1):
                    end = start + required_length
                    unique = 0
                    all_k = True
                    for c in range(26):
                        cnt = prefix[c][end] - prefix[c][start]
                        if cnt > 0:
                            unique += 1
                            if cnt != k:
                                all_k = False
                                break
                    if unique == m and all_k:
                        total += 1
        
        return total