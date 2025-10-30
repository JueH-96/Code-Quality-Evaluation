class Solution:
    def findLatestTime(self, s: str) -> str:
        ans = ""
        for i in range(len(s)):
            if s[i] == '?':
                if i == 0:
                    if s[1] == '?' or int(s[1]) <= 9:
                        ans += '2'
                        if s[1] == '?':
                            if ans[0] == '2':
                                ans += '3'
                            else:
                                ans += '9'
                    elif int(s[1]) >= 4:
                        ans += '1'
                        if s[1] == '?':
                            ans += '9'
                    else:
                        ans += '1'
                elif i == 1:
                    if ans[0] == '2':
                        ans += '3'
                    else:
                        ans += '9'

                elif i == 3:
                    ans += '5'
                elif i == 4:
                    ans += '9'
            else:
                ans += s[i]

        if int(ans[0:2]) > 11 and ans[0] == '2':
            ans = list(ans)
            ans[0] = '1'
            ans = "".join(ans)
        return ans