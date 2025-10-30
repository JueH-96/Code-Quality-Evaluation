class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        pre, post = p.split('*')
        if not pre and not post:
            return True
        elif not pre:
            return post in s
        elif not post:
            return pre in s
        else:
            len_pre = len(pre)
            len_post = len(post)
            len_s = len(s)
            for i in range(len_s - len_pre + 1):
                if s[i:i+len_pre] == pre:
                    start_j = i + len_pre
                    end_j = len_s - len_post + 1
                    for j in range(start_j, end_j):
                        if s[j:j+len_post] == post:
                            return True
            return False