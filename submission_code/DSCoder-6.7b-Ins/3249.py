class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        from collections import deque
        nums = [bin(n)[2:] for n in nums]
        max_len = max([len(num) for num in nums])
        nums = [(max_len - len(num))*'0' + num for num in nums]
        nums = [list(map(int, num)) for num in nums]
        nums = [sum([num[i]*(2**i) for i in range(max_len)]) for num in nums]
        k = bin(k)[2:]
        k = (max_len - len(k))*'0' + k
        k = list(map(int, k))
        k = sum([k[i]*(2**i) for i in range(max_len)])
        queue = deque([(0, 0, 0)])
        visited = {0}
        while queue:
            cur, ops, xor = queue.popleft()
            if xor == k:
                return ops
            for num in nums:
                nxt = cur ^ num
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, ops+1, xor+num))
        return -1