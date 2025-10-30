from itertools import product
import math

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(math.sqrt(x)) + 1):
                if x % i == 0:
                    return False
            return True

        def get_digits(x):
            return [int(d) for d in str(x)]

        def to_number(digits):
            return int(''.join(map(str, digits)))

        def is_valid(digits):
            num = to_number(digits)
            return not is_prime(num)

        def get_neighbors(digits):
            neighbors = []
            for i in range(len(digits)):
                if digits[i] != 9:
                    new_digits = digits.copy()
                    new_digits[i] += 1
                    if is_valid(new_digits):
                        neighbors.append(new_digits)
                if digits[i] != 0:
                    new_digits = digits.copy()
                    new_digits[i] -= 1
                    if is_valid(new_digits):
                        neighbors.append(new_digits)
            return neighbors

        start_digits = get_digits(n)
        target_digits = get_digits(m)

        if not is_valid(start_digits) or not is_valid(target_digits):
            return -1

        from collections import deque
        queue = deque()
        queue.append((start_digits, 0))
        visited = set()
        visited.add(tuple(start_digits))

        while queue:
            current_digits, cost = queue.popleft()
            if current_digits == target_digits:
                return cost + to_number(current_digits)
            for neighbor in get_neighbors(current_digits):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    queue.append((neighbor, cost + to_number(current_digits)))

        return -1