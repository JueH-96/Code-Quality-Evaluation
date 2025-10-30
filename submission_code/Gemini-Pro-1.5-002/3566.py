class Solution:
    def stringSequence(self, target: str) -> List[str]:
        ans = []
        s = ""
        for char in target:
            while len(s) < len(target) and s + 'a' * (len(target) - len(s)) != target[:len(s)] + 'a' * (len(target) - len(s)):
                s += 'a'
                ans.append(s)

            while len(s) > 0 and s != target[:len(s)]:
                s = s[:-1] + chr(ord(s[-1]) + 1)
                ans.append(s)
            if len(s) < len(target):
                s += 'a'
                ans.append(s)
        return ans