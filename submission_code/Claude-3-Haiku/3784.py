class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        answer = []
        
        for i in range(n):
            remaining_words = words[:i] + words[i+1:]
            if len(remaining_words) < k:
                answer.append(0)
                continue
            
            common_prefix = self.findLongestCommonPrefix(remaining_words[:k])
            answer.append(len(common_prefix))
        
        return answer
    
    def findLongestCommonPrefix(self, words: List[str]) -> str:
        if not words:
            return ""
        
        prefix = words[0]
        for i in range(1, len(words)):
            while prefix and not words[i].startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        
        return prefix