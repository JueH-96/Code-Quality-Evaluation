class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        if not word:
            return 0
        
        # Split the word into segments where each segment has adjacent characters with <= 2 difference
        segments = []
        start = 0
        for i in range(1, len(word)):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])
        
        total = 0
        
        for seg in segments:
            n = len(seg)
            for m in range(1, 27):
                s = m * k
                if s > n:
                    continue
                
                count = [0] * 26
                exact_k = 0
                distinct = 0
                
                # Initialize the first window
                for i in range(s):
                    c = seg[i]
                    idx = ord(c) - ord('a')
                    prev = count[idx]
                    count[idx] += 1
                    if prev == 0:
                        distinct += 1
                    if prev == k:
                        exact_k -= 1
                    if count[idx] == k:
                        exact_k += 1
                
                if exact_k == m and distinct == m:
                    total += 1
                
                # Slide the window
                for i in range(s, len(seg)):
                    # Remove the left character
                    left_char = seg[i - s]
                    left_idx = ord(left_char) - ord('a')
                    prev_left = count[left_idx]
                    
                    if prev_left == k:
                        exact_k -= 1
                    count[left_idx] -= 1
                    if count[left_idx] == k:
                        exact_k += 1
                    if count[left_idx] == 0:
                        distinct -= 1
                    
                    # Add the right character
                    right_char = seg[i]
                    right_idx = ord(right_char) - ord('a')
                    prev_right = count[right_idx]
                    
                    if prev_right == k:
                        exact_k -= 1
                    count[right_idx] += 1
                    if count[right_idx] == k:
                        exact_k += 1
                    if prev_right == 0:
                        distinct += 1
                    
                    # Check if valid
                    if exact_k == m and distinct == m:
                        total += 1
        
        return total