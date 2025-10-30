class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from heapq import heappush, heappop
        
        # Count forced occurrences and number of '?'
        forced_count = [0]*26
        question_positions = []
        for ch in s:
            if ch == '?':
                question_positions.append(ch)
            else:
                forced_count[ord(ch) - ord('a')] += 1
        
        unknown = len(question_positions)
        
        # Build a min-heap of (current_count, letter_index)
        # current_count = forced_count[letter_index] initially
        heap = []
        for i in range(26):
            heappush(heap, (forced_count[i], i))
        
        # Distribute "unknown" question marks among letters to minimize
        # sum of k_c*(k_c - 1)/2, by always picking the letter with the smallest count so far
        assigned_count = [0]*26
        for _ in range(unknown):
            curr_count, letter_idx = heappop(heap)
            assigned_count[letter_idx] += 1
            new_count = curr_count + 1
            heappush(heap, (new_count, letter_idx))
        
        # Build a sorted list of letters to place into the '?' positions
        # Placing them in ascending order yields the lexicographically smallest solution
        fill_letters = []
        for i in range(26):
            fill_letters.extend([chr(ord('a') + i)] * assigned_count[i])
        
        # Fill the '?'s in the original string with letters in fill_letters
        s_list = list(s)
        fill_index = 0
        for i in range(len(s_list)):
            if s_list[i] == '?':
                s_list[i] = fill_letters[fill_index]
                fill_index += 1
        
        return "".join(s_list)