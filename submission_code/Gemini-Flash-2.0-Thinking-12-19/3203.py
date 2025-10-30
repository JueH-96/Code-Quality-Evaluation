from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        results = []
        for query in queries:
            a_i, b_i, c_i, d_i = query
            count1 = Counter(s[a_i:b_i+1])
            count2 = Counter(s[c_i:d_i+1])
            needed_chars_count = Counter()
            possible = True
            for j in range(n // 2):
                k = n - 1 - j
                is_j_in_range1 = (a_i <= j <= b_i)
                is_k_in_range2 = (c_i <= k <= d_i)
                if (not is_j_in_range1) and (not is_k_in_range2):
                    if s[j] != s[k]:
                        possible = False
                        break
                elif is_j_in_range1 and (not is_k_in_range2):
                    char_to_find = s[k]
                    needed_chars_count[char_to_find] += 1
                elif (not is_j_in_range1) and is_k_in_range2:
                    char_to_find = s[j]
                    needed_chars_count[char_to_find] += 1
            if not possible:
                results.append(False)
                continue
            
            possible_rearrangement = True
            for char_name, required_count in needed_chars_count.items():
                available_count_in_range1 = count1.get(char_name, 0)
                available_count_in_range2 = count2.get(char_name, 0)
                total_available_count = available_count_in_range1 + available_count_in_range2
                if total_available_count < required_count:
                    possible_rearrangement = False
                    break
                
            results.append(possible_rearrangement)
            
        return results