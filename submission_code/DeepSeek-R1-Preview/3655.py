class Solution:
    def __init__(self):
        self.sieve = self.compute_sieve(9999)
    
    def compute_sieve(self, max_num):
        sieve = [True] * (max_num + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_num ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i : max_num+1 : i] = [False] * len(sieve[i*i : max_num+1 : i])
        return sieve
    
    def minOperations(self, n: int, m: int) -> int:
        sieve = self.sieve
        if sieve[n] or sieve[m]:
            return -1
        if n == m:
            return n
        
        digit_count = len(str(n))
        n_str = str(n).zfill(digit_count)
        m_str = str(m).zfill(digit_count)
        
        import heapq
        heap = []
        heapq.heappush(heap, (n, n_str))
        min_cost = {n_str: n}
        
        while heap:
            current_cost, current = heapq.heappop(heap)
            if current == m_str:
                return current_cost
            for i in range(digit_count):
                current_digit = current[i]
                if current_digit != '9':
                    new_digit = str(int(current_digit) + 1)
                    new_state = current[:i] + new_digit + current[i+1:]
                    num = int(new_state)
                    if not sieve[num]:
                        new_cost = current_cost + num
                        if new_state not in min_cost or new_cost < min_cost[new_state]:
                            min_cost[new_state] = new_cost
                            heapq.heappush(heap, (new_cost, new_state))
                if current_digit != '0':
                    new_digit = str(int(current_digit) - 1)
                    new_state = current[:i] + new_digit + current[i+1:]
                    num = int(new_state)
                    if not sieve[num]:
                        new_cost = current_cost + num
                        if new_state not in min_cost or new_cost < min_cost[new_state]:
                            min_cost[new_state] = new_cost
                            heapq.heappush(heap, (new_cost, new_state))
        return -1