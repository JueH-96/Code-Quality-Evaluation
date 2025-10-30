class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        mod = 10**9 + 7
        state = [1] * 26
        
        for _ in range(t):
            new_state = [0] * 26
            for i in range(25):
                new_state[i] = state[i+1]
            new_state[25] = (state[0] + state[1]) % mod
            state = new_state
        
        total = 0
        for char in s:
            idx = ord(char) - ord('a')
            total = (total + state[idx]) % mod
        return total