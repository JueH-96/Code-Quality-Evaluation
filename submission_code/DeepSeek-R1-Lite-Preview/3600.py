class Solution:
    def kthCharacter(self, k: int) -> str:
        # Find the smallest s such that 2^s >= k
        s = 0
        length = 1
        while length < k:
            length <<= 1
            s += 1
        
        shift_count = 0
        current_length = length
        current_s = s
        target_k = k
        
        while current_s > 0:
            half = current_length >> 1  # equivalent to 2^(current_s - 1)
            if target_k > half:
                shift_count += 1
                target_k -= half
            current_s -= 1
            current_length = half
        
        # Calculate the final character
        final_char = chr(97 + (shift_count % 26))
        return final_char