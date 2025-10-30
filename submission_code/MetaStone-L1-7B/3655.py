def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_next(x):
    s = list(str(x))
    next_nums = []
    for i in range(len(s)):
        current = int(s[i])
        if i == 0:
            if current < 9:
                new_digit = current + 1
                new_s = s[:i] + [str(new_digit)] + s[i+1:]
                next_num = int(''.join(new_s))
                if len(str(next_num)) == len(str(x)):
                    next_nums.append(next_num)
            if current > 0:
                new_digit = current - 1
                new_s = s[:i] + [str(new_digit)] + s[i+1:]
                next_num = int(''.join(new_s))
                if len(str(next_num)) == len(str(x)):
                    next_nums.append(next_num)
        else:
            if current < 9:
                new_digit = current + 1
                new_s = s[:i] + [str(new_digit)] + s[i+1:]
                next_num = int(''.join(new_s))
                if len(str(next_num)) == len(str(x)):
                    next_nums.append(next_num)
            if current > 0:
                new_digit = current - 1
                new_s = s[:i] + [str(new_digit)] + s[i+1:]
                next_num = int(''.join(new_s))
                if len(str(next_num)) == len(str(x)):
                    next_nums.append(next_num)
    unique_next = list(set(next_nums))
    return unique_next

class Solution:
    def min_operations(self, n: int, m: int) -> int:
        if not is_prime(m):
            return -1
        if n == m:
            return n
        heap = []
        heapq.heappush(heap, (n, n))
        visited = {}
        visited[n] = n
        while heap:
            current_sum, x = heapq.heappop(heap)
            if x == m:
                return current_sum
            for next_num in generate_next(x):
                if is_prime(next_num):
                    continue
                if next_num in visited:
                    if visited[next_num] <= current_sum + next_num:
                        continue
                visited[next_num] = current_sum + next_num
                heapq.heappush(heap, (current_sum + next_num, next_num))
        return -1