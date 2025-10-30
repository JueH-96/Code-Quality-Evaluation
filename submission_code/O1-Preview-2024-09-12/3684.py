class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        pre, post = p.split('*')
        len_pre = len(pre)
        len_post = len(post)
        for i in range(len(s) - len_pre + 1):
            if s[i:i+len_pre] == pre:
                min_j = i + len_pre + len_post
                for j in range(min_j, len(s)+1):
                    if s[j - len_post:j] == post:
                        return True
        return False