class Solution:
    def maximumLength(self, s: str) -> int:
        # Edge case handling (though constraints say s.length >=3)
        if not s:
            return -1
        
        # Step 1: Find all runs of consecutive characters
        runs = []
        prev_char = s[0]
        count = 1
        for c in s[1:]:
            if c == prev_char:
                count += 1
            else:
                runs.append((prev_char, count))
                prev_char = c
                count = 1
        runs.append((prev_char, count))
        
        # Step 2: Group runs by character
        from collections import defaultdict
        groups = defaultdict(list)
        for c, l in runs:
            groups[c].append(l)
        
        # Step 3: Build frequency map for each character and length
        freq = defaultdict(lambda: defaultdict(int))
        for c in groups:
            for run_length in groups[c]:
                for l in range(1, run_length + 1):
                    cnt = run_length - l + 1
                    freq[c][l] += cnt
        
        # Step 4: Determine the maximum length with count >=3
        max_len = -1
        for c in freq:
            # Sort lengths in descending order to find the maximum first
            lengths = sorted(freq[c].keys(), reverse=True)
            for l in lengths:
                if freq[c][l] >= 3:
                    if l > max_len:
                        max_len = l
                    break  # Break early after finding the largest possible length for this c
        
        return max_len if max_len != -1 else -1