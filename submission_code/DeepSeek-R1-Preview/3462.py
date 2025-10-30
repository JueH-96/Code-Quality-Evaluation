class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        current_parity = 0
        parity_count = {0: 1}
        total_odd = 0
        
        for char in s:
            if char in vowels:
                current_parity ^= 1  # Toggle parity if the character is a vowel
            # The number of substrings ending here with odd vowels is the count of the opposite parity
            opposite_parity = current_parity ^ 1
            total_odd += parity_count.get(opposite_parity, 0)
            # Update the count for the current parity
            parity_count[current_parity] = parity_count.get(current_parity, 0) + 1
        
        return total_odd > 0