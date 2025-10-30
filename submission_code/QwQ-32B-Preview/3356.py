from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        # Helper function to generate all substrings of a string
        def get_substrings(s):
            substrings = set()
            n = len(s)
            for i in range(n):
                for j in range(i + 1, n + 1):
                    substrings.add(s[i:j])
            return substrings
        
        # Step 1: Generate all substrings for each string
        all_substrings = [get_substrings(s) for s in arr]
        
        # Step 2: Create a dictionary mapping substrings to the indices of strings they appear in
        substring_dict = {}
        for idx, substrings in enumerate(all_substrings):
            for substring in substrings:
                if substring in substring_dict:
                    substring_dict[substring].append(idx)
                else:
                    substring_dict[substring] = [idx]
        
        # Step 3: For each string, find substrings unique to it
        answer = []
        for i in range(len(arr)):
            unique_substrings = [substring for substring, indices in substring_dict.items() if indices == [i]]
            
            # Step 4: Find the shortest lexicographically smallest unique substring
            if unique_substrings:
                # Find the minimum length
                min_length = min(len(sub) for sub in unique_substrings)
                # Filter substrings with that length
                candidates = [sub for sub in unique_substrings if len(sub) == min_length]
                # Choose the lexicographically smallest
                answer.append(min(candidates))
            else:
                answer.append("")
        
        return answer