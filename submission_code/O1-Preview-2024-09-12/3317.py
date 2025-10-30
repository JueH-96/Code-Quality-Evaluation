class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        from collections import Counter
        
        counts = Counter()
        for word in words:
            counts.update(word)
        
        total_pairs_avail = sum(count // 2 for count in counts.values())
        total_singles_avail = sum(count % 2 for count in counts.values())
        
        words_info = []
        for word in words:
            length = len(word)
            P_i = length // 2  # Number of pairs needed
            C_i = length % 2   # 1 if odd length (center letter needed), 0 otherwise
            words_info.append((P_i, C_i, length))
        
        ans = 0
        # Process words with odd lengths
        words_odd = [info for info in words_info if info[1] == 1]
        words_odd.sort(key=lambda x: x[0])  # Sort by P_i (number of pairs needed)
        for P_i, C_i, _ in words_odd:
            if total_pairs_avail >= P_i and total_singles_avail >= 1:
                total_pairs_avail -= P_i
                total_singles_avail -= 1
                ans += 1
            else:
                break  # Cannot assign more odd-length palindromic words
        
        # Process words with even lengths
        words_even = [info for info in words_info if info[1] == 0]
        words_even.sort(key=lambda x: x[0])  # Sort by P_i (number of pairs needed)
        for P_i, C_i, _ in words_even:
            if total_pairs_avail >= P_i:
                total_pairs_avail -= P_i
                ans += 1
            else:
                break  # Cannot assign more even-length palindromic words
        
        return ans