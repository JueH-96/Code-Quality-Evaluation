from collections import defaultdict

class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        
        # Calculate the number of vowels and consonants up to each index
        v_count, c_count = [0], [0]
        for ch in s:
            v_count.append(v_count[-1] + (ch in vowels))
            c_count.append(c_count[-1] + (ch in consonants))
        
        # Calculate all possible pairs of lengths (v, c) such that v == c and (v * c) % k == 0
        combinations = set()
        for v in range(1, 21):  # Maximum of 20 vowels or consonants within a substring of length 1000
            for c in range(v, 21):  # Using symmetry to reduce computations
                if v == c and (v * c) % k == 0:
                    combinations.add((v, c))
        
       ansa = defaultdict(int)
        result = 0
        # Iterate over all possible substrings using the calculated prefix counts
        for i in range(len(s)):
            sub_v = v_count[i + 1] - v_count[i]
            sub_c = c_count[i + 1] - c_count[i]
            if (sub_v, sub_c) in combinations:
                result += 1  # Count the substring itself
            for v, c in combinations:
                v_target = v + sub_v
                c_target = c + sub_c
                if v_target >= sub_v and c_target >= sub_c:
                    ansa[(v_target, c_target)] += 1  # Increment the count for a substring extended from i
                if v_target >= sub_v and c_target >= sub_c and (v_target, c_target) in ansa:
                    result += ansa[(v_target, c_target)]  # Increment the result by the count of substrings that can be extended
        
        return result