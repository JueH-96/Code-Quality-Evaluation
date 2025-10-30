class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        stack, ans = deque(), []
        
        for word in words:
            if word == "prev":
                ans.append(-1) if not stack else stack.pop()
            else:
                stack.append(int(word))
                
        if not stack: return ans
        stack = deque(stack)
        l = len(stack) - 1
        for i in range(len(ans)):
            if ans[i] == -1: continue
            ans[i] = stack[l]
            if i >= l: break
            l -= 1
        return ans