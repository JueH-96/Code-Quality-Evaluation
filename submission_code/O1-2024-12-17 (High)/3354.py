class Solution:
    def minimizeStringValue(self, s: str) -> str:
        import heapq
        
        # Convert s to a list for easy in-place modifications
        chars = list(s)
        
        # Count initial frequencies of given letters (ignore '?')
        freq = [0]*26
        for ch in chars:
            if ch != '?':
                freq[ord(ch) - ord('a')] += 1
        
        # Create a min-heap (frequency, letter_index)
        # so that letters with smaller frequency come out first.
        # Ties in frequency are broken by smaller letter_index => lexicographically smaller.
        pq = [(freq[i], i) for i in range(26)]
        heapq.heapify(pq)
        
        # Replace each '?' one by one
        for i in range(len(chars)):
            if chars[i] == '?':
                f, letter_id = heapq.heappop(pq)
                chars[i] = chr(letter_id + ord('a'))
                # Increment the frequency of the chosen letter
                heapq.heappush(pq, (f + 1, letter_id))
        
        # Return the modified string
        return "".join(chars)