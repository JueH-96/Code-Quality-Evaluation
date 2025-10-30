import heapq

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Get initial counts of non-'?' characters and the total number of '?'s.
        counts = [0] * 26
        q_count = 0
        for char in s:
            if char == '?':
                q_count += 1
            else:
                counts[ord(char) - ord('a')] += 1
        
        # If there are no '?'s, the string is already determined.
        if q_count == 0:
            return s

        # Use a min-priority queue to greedily find the replacement characters.
        # The priority queue stores (count, character) tuples. When counts are
        # equal, Python's heapq tie-breaks using the second element (the character),
        # which correctly implements the lexicographical preference.
        pq = []
        for i in range(26):
            heapq.heappush(pq, (counts[i], chr(ord('a') + i)))

        # Determine the q_count replacement characters by repeatedly picking
        # the character with the minimum current count.
        replacement_chars = []
        for _ in range(q_count):
            count, char_to_use = heapq.heappop(pq)
            replacement_chars.append(char_to_use)
            # Push the character back with an incremented count for subsequent decisions.
            heapq.heappush(pq, (count + 1, char_to_use))

        # Sort the replacement characters. This is crucial for constructing the
        # lexicographically smallest output string by filling '?'s from left
        # to right with the smallest available characters.
        replacement_chars.sort()

        # Build the final string by replacing '?'s in order.
        result_list = []
        fill_idx = 0
        for char in s:
            if char == '?':
                result_list.append(replacement_chars[fill_idx])
                fill_idx += 1
            else:
                result_list.append(char)
        
        return "".join(result_list)