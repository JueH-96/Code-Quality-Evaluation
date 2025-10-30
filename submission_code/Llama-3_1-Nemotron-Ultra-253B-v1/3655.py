class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        if is_prime(n) or is_prime(m):
            return -1
        if n == m:
            return n
        
        heap = []
        heapq.heappush(heap, (n, n))
        visited = {n: n}
        
        while heap:
            total_cost, current = heapq.heappop(heap)
            if current == m:
                return total_cost
            if total_cost > visited.get(current, float('inf')):
                continue
            
            digits = list(map(int, str(current)))
            num_digits = len(digits)
            for i in range(num_digits):
                original = digits[i]
                for delta in [-1, 1]:
                    new_digit = original + delta
                    if i == 0:
                        if (delta == -1 and new_digit < 1) or (delta == 1 and new_digit > 9):
                            continue
                    else:
                        if (delta == -1 and new_digit < 0) or (delta == 1 and new_digit > 9):
                            continue
                    new_digits = digits.copy()
                    new_digits[i] = new_digit
                    if new_digits[0] == 0:
                        continue
                    new_num = int(''.join(map(str, new_digits)))
                    if is_prime(new_num):
                        continue
                    new_cost = total_cost + new_num
                    if new_num in visited and visited[new_num] <= new_cost:
                        continue
                    visited[new_num] = new_cost
                    heapq.heappush(heap, (new_cost, new_num))
        
        return -1