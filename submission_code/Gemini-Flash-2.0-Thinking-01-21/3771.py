class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        first = [-1] * 26
        last = [-1] * 26
        for i in range(n):
            char_idx = ord(s[i]) - ord('a')
            if first[char_idx] == -1:
                first[char_idx] = i
            last[char_idx] = i

        count = 0
        block_start = 0
        max_last_in_block = -1

        i = 0
        while i < n:
            char_i_idx = ord(s[i]) - ord('a')

            # If character s[i] first appeared before the start of the current block,
            # it means s[i] belongs to an earlier special segment.
            # The current segment attempt starting at block_start is invalid.
            # We must abandon this block and start searching for the next special substring
            # after the segment containing s[i] ends, which is last[char_i_idx].
            if first[char_i_idx] < block_start:
                # Jump the pointer 'i' past the range of the problematic character.
                # The search for the next potential block starts after last[char_i_idx].
                i = last[char_i_idx] + 1
                block_start = i  # The new block starts at the jumped 'i'.
                max_last_in_block = -1 # Reset max_last for the new block.
                continue # Go to the next iteration with the new 'i'.

            # If first[char_i_idx] >= block_start, the character is compatible with the current block.
            # Update the maximum required end for the current block.
            if i == block_start:
                 # This is the first character of a potentially new block.
                 max_last_in_block = last[char_i_idx]
            else:
                 # Update max_last based on the current character.
                 max_last_in_block = max(max_last_in_block, last[char_i_idx])

            # If the current index 'i' reaches the maximum required end for the block,
            # it means we have found a valid special segment s[block_start : i+1].
            if i == max_last_in_block:
                # Check if this segment is not the entire string.
                # The segment is the entire string if block_start is 0 and i is n-1.
                if not (block_start == 0 and i == n - 1):
                    count += 1

                # The next block search starts immediately after this one ends.
                block_start = i + 1
                # max_last_in_block will be reset when i == block_start in the next iteration

            # Move to the next index in the string, unless 'i' was jumped.
            i += 1

        return count >= k