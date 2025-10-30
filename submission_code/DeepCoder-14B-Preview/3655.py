import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        if is_prime(n) or is_prime(m):
            return -1
        
        if n == m:
            return n
        
        required_digits = len(str(n))
        heap = []
        heapq.heappush(heap, (n, n))
        visited = {n: n}
        
        while heap:
            current_sum, current_num = heapq.heappop(heap)
            
            if current_num == m:
                return current_sum
            
            s = str(current_num)
            if len(s) != required_digits:
                continue
            
            for i in range(len(s)):
                d = int(s[i])
                
                # Increase the digit
                if d < 9:
                    new_d = d + 1
                    new_s = s[:i] + str(new_d) + s[i+1:]
                    new_num = int(new_s)
                    if len(str(new_num)) == required_digits:
                        if not is_prime(new_num):
                            new_sum = current_sum + new_num
                            if new_num not in visited or new_sum < visited.get(new_num, float('inf')):
                                visited[new_num] = new_sum
                                heapq.heappush(heap, (new_sum, new_num))
                
                # Decrease the digit
                if d > 0:
                    new_d = d - 1
                    new_s = s[:i] + str(new_d) + s[i+1:]
                    new_num = int(new_s)
                    if len(str(new_num)) == required_digits:
                        if not is_prime(new_num):
                            new_sum = current_sum + new_num
                            if new_num not in visited or new_sum < visited.get(new_num, float('inf')):
                                visited[new_num] = new_sum
                                heapq.heappush(heap, (new_sum, new_num))
        
        return -1