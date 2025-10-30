class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        # Precompute the frequency of each character in the first and second half
        freq_first = [0] * 26
        freq_second = [0] * 26
        for i in range(half):
            freq_first[ord(s[i]) - ord('a')] += 1
        for i in range(half, n):
            freq_second[ord(s[i]) - ord('a')] += 1
        # Check if the total frequency of each character is even
        for i in range(26):
            if (freq_first[i] + freq_second[i]) % 2 != 0:
                return [False] * len(queries)
        # Precompute prefix sums for the first and second half
        prefix_first = [[0] * 26 for _ in range(half + 1)]
        prefix_second = [[0] * 26 for _ in range(half + 1)]
        for i in range(half):
            for j in range(26):
                prefix_first[i+1][j] = prefix_first[i][j]
            prefix_first[i+1][ord(s[i]) - ord('a')] += 1
        for i in range(half, n):
            for j in range(26):
                prefix_second[i - half + 1][j] = prefix_second[i - half][j]
            prefix_second[i - half + 1][ord(s[i]) - ord('a')] += 1
        # Function to get the frequency of characters in a range
        def get_freq(prefix, l, r):
            res = [0] * 26
            for i in range(26):
                res[i] = prefix[r][i] - prefix[l][i]
            return res
        # Process each query
        answer = []
        for q in queries:
            a, b, c, d = q
            # Convert to 0-based indices for the second half
            c -= half
            d -= half
            # Get the frequency of characters in the allowed ranges
            freq_a_b = get_freq(prefix_first, a, b+1)
            freq_c_d = get_freq(prefix_second, c, d+1)
            # Calculate the frequency of characters in the non-allowed ranges
            # For the first half: 0..a-1 and b+1..half-1
            freq_first_non = [0] * 26
            for i in range(26):
                freq_first_non[i] = freq_first[i] - freq_a_b[i]
            # For the second half: 0..c-1 and d+1..half-1
            freq_second_non = [0] * 26
            for i in range(26):
                freq_second_non[i] = freq_second[i] - freq_c_d[i]
            # Check if the non-allowed ranges can form a palindrome
            # The non-allowed ranges must have the same frequency for each character
            possible = True
            for i in range(26):
                if freq_first_non[i] != freq_second_non[i]:
                    possible = False
                    break
            answer.append(possible)
        return answer