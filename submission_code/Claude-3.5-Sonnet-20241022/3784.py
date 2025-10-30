class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        def get_lcp_length(s1: str, s2: str) -> int:
            # Get length of longest common prefix between two strings
            i = 0
            while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
                i += 1
            return i
        
        def find_max_k_lcp(word_list: List[str], k: int) -> int:
            if len(word_list) < k:
                return 0
                
            # For each word, find all possible LCPs with other words
            word_lcps = {}  # word -> list of LCP lengths with other words
            for i in range(len(word_list)):
                if word_list[i] not in word_lcps:
                    word_lcps[word_list[i]] = []
                    for j in range(len(word_list)):
                        if i != j:
                            lcp_len = get_lcp_length(word_list[i], word_list[j])
                            word_lcps[word_list[i]].append(lcp_len)
            
            # For each unique word, sort its LCPs and check if we can form k occurrences
            max_lcp = 0
            for word in word_lcps:
                count = word_list.count(word)
                lcps = sorted(word_lcps[word], reverse=True)
                
                # If we have k or more identical words
                if count >= k:
                    max_lcp = max(max_lcp, len(word))
                    continue
                
                # We need (k - count) more words with LCP
                needed = k - count
                if needed > len(lcps):
                    continue
                    
                max_lcp = max(max_lcp, lcps[needed - 1])
                
            return max_lcp
        
        n = len(words)
        answer = []
        
        # For each index
        for i in range(n):
            # Create new list without the current word
            remaining_words = words[:i] + words[i+1:]
            # Find max LCP among any k strings
            max_lcp = find_max_k_lcp(remaining_words, k)
            answer.append(max_lcp)
            
        return answer