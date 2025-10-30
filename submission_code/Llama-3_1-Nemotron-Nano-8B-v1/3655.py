class Solution:
    def __init__(self):
        max_num = 10**4
        self.is_prime = [True] * (max_num + 1)
        self.is_prime[0] = self.is_prime[1] = False
        for i in range(2, int(max_num ** 0.5) + 1):
            if self.is_prime[i]:
                for j in range(i * i, max_num + 1, i):
                    self.is_prime[j] = False

    def minOperations(self, n: int, m: int) -> int:
        if self.is_prime[n] or self.is_prime[m]:
            return -1
        if n == m:
            return n
        
        import heapq
        heap = []
        heapq.heappush(heap, (n, n))
        visited = {n: n}
        
        while heap:
            current_sum, current_num = heapq.heappop(heap)
            
            if current_num == m:
                return current_sum
            
            if current_sum > visited.get(current_num, float('inf')):
                continue
            
            s = str(current_num)
            next_nums = set()
            for i in range(len(s)):
                digit = int(s[i])
                # Increase the digit
                if digit < 9:
                    new_s = s[:i] + str(digit + 1) + s[i+1:]
                    new_num = int(new_s)
                    if not self.is_prime[new_num]:
                        next_nums.add(new_num)
                # Decrease the digit
                if digit > 0:
                    new_s = s[:i] + str(digit - 1) + s[i+1:]
                    new_num = int(new_s)
                    if not self.is_prime[new_num]:
                        next_nums.add(new_num)
            
            for next_num in next_nums:
                new_sum = current_sum + next_num
                if next_num not in visited or new_sum < visited[next_num]:
                    visited[next_num] = new_sum
                    heapq.heappush(heap, (new_sum, next_num))
        
        return -1