class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(x):
            if x < 2:
                return False
            if x == 2:
                return True
            if x % 2 == 0:
                return False
            for i in range(3, int(x**0.5) + 1, 2):
                if x % i == 0:
                    return False
            return True
        
        # Initial checks
        if is_prime(n) or is_prime(m):
            return -1
        if n == m:
            return n
        
        d = len(str(n))  # Number of digits
        
        import heapq
        heap = []
        heapq.heappush(heap, (n, n))  # (cost, current_number)
        distance = {n: n}
        
        while heap:
            current_cost, current_num = heapq.heappop(heap)
            
            if current_num == m:
                return current_cost
            
            # Skip if a cheaper path was already found
            if current_cost > distance.get(current_num, float('inf')):
                continue
            
            # Generate all possible next numbers
            s = list(str(current_num))
            for i in range(len(s)):
                # Increment current digit
                if s[i] < '9':
                    new_s = s.copy()
                    new_s[i] = chr(ord(new_s[i]) + 1)
                    new_num = int(''.join(new_s))
                    if not is_prime(new_num):
                        new_cost = current_cost + new_num
                        if new_num not in distance or new_cost < distance[new_num]:
                            distance[new_num] = new_cost
                            heapq.heappush(heap, (new_cost, new_num))
                
                # Decrement current digit
                if i == 0:
                    if s[i] > '1':
                        new_s = s.copy()
                        new_s[i] = chr(ord(new_s[i]) - 1)
                        new_num = int(''.join(new_s))
                        if not is_prime(new_num):
                            new_cost = current_cost + new_num
                            if new_num not in distance or new_cost < distance[new_num]:
                                distance[new_num] = new_cost
                                heapq.heappush(heap, (new_cost, new_num))
                else:
                    if s[i] > '0':
                        new_s = s.copy()
                        new_s[i] = chr(ord(new_s[i]) - 1)
                        new_num = int(''.join(new_s))
                        if not is_prime(new_num):
                            new_cost = current_cost + new_num
                            if new_num not in distance or new_cost < distance[new_num]:
                                distance[new_num] = new_cost
                                heapq.heappush(heap, (new_cost, new_num))
        
        return -1