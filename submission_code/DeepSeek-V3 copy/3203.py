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
        # Precompute the prefix sums for the first and second half
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
        for query in queries:
            a, b, c, d = query
            # Convert to 0-based indices for the second half
            c -= half
            d -= half
            # Get the frequency of characters in the first and second half ranges
            freq_a_b = get_freq(prefix_first, a, b+1)
            freq_c_d = get_freq(prefix_second, c, d+1)
            # Calculate the frequency of characters in the entire first and second half
            total_first = get_freq(prefix_first, 0, half)
            total_second = get_freq(prefix_second, 0, half)
            # Calculate the frequency of characters outside the ranges
            freq_outside_first = [total_first[i] - freq_a_b[i] for i in range(26)]
            freq_outside_second = [total_second[i] - freq_c_d[i] for i in range(26)]
            # Check if the outside frequencies are the same
            if freq_outside_first != freq_outside_second:
                answer.append(False)
                continue
            # Check if the sum of frequencies in the ranges is the same
            if sum(freq_a_b) != sum(freq_c_d):
                answer.append(False)
                continue
            # Check if the frequencies in the ranges can be rearranged to match
            # For each character, the sum of frequencies in the ranges should be even
            # Because we need to pair them
            possible = True
            for i in range(26):
                if (freq_a_b[i] + freq_c_d[i]) % 2 != 0:
                    possible = False
                    break
            answer.append(possible)
        return answer