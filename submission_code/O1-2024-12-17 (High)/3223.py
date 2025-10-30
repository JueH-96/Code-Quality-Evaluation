class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        # First, split 'word' into segments where any two adjacent characters
        # differ by more than 2. No valid "complete" substring can cross these boundaries.
        segments = []
        start = 0
        for i in range(len(word) - 1):
            if abs(ord(word[i+1]) - ord(word[i])) > 2:
                segments.append(word[start:i+1])
                start = i + 1
        # Append the last segment
        segments.append(word[start:])

        def count_for_segment(seg):
            """
            Counts how many substrings within the given segment 'seg' have each
            distinct character appearing exactly k times. Since 'seg' itself is
            guaranteed not to break the adjacency condition, any contiguous
            substring of 'seg' also satisfies the adjacency condition.
            """
            n = len(seg)
            ans = 0
            # We'll try d distinct characters (1..26). A valid substring must have length = d*k.
            # Then use a fixed-size sliding window of length d*k to check if each
            # distinct character in that window occurs exactly k times.
            for d in range(1, 27):
                length_req = d * k
                if length_req > n:
                    break

                freq = [0] * 26  # frequency array for 'a'..'z'
                distinct_count = 0
                freq_k_count = 0  # how many characters have frequency == k

                # Initialize the first window of size length_req
                for i in range(length_req):
                    c = ord(seg[i]) - ord('a')
                    if freq[c] == 0:
                        distinct_count += 1
                    freq[c] += 1
                    if freq[c] == k:
                        freq_k_count += 1
                    elif freq[c] == k + 1:
                        freq_k_count -= 1

                # Check if the first window is valid
                if distinct_count == d and freq_k_count == d:
                    ans += 1

                # Slide the window through the segment
                for right in range(length_req, n):
                    left_char = ord(seg[right - length_req]) - ord('a')
                    # Remove the left-most char from the window
                    if freq[left_char] == k:
                        freq_k_count -= 1
                    elif freq[left_char] == k + 1:
                        freq_k_count += 1
                    freq[left_char] -= 1
                    if freq[left_char] == 0:
                        distinct_count -= 1

                    # Add the new right-most char to the window
                    new_char = ord(seg[right]) - ord('a')
                    if freq[new_char] == 0:
                        distinct_count += 1
                    freq[new_char] += 1
                    if freq[new_char] == k:
                        freq_k_count += 1
                    elif freq[new_char] == k + 1:
                        freq_k_count -= 1

                    # Check validity after the shift
                    if distinct_count == d and freq_k_count == d:
                        ans += 1

            return ans

        # Sum valid substrings over all segments
        total_complete = 0
        for seg in segments:
            total_complete += count_for_segment(seg)

        return total_complete