class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        
        def helper(s, pat):
            pat_len = len(pat)
            curr_pos = Answ = -1
            pos_list = []
            while True:
                curr_pos = s.find(pat, curr_pos + 1)
                if curr_pos == -1: break
                pos_list.append(curr_pos)
            return pos_list
        
        posA = helper(s, a)
        posB = helper(s, b)
        
        Answer = []

        for i in range(len(posA)):
            for j in range(len(posB)):
                if abs(posA[i] - posB[j]) <= k:
                    Answer.append(posA[i])
        
        return Answer