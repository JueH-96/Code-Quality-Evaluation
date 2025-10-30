class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num <= 1:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        result = []
        for x in range(2, n // 2 + 1):
            y = n - x
            if y < x:
                break
            if is_prime(x) and is_prime(y):
                result.append([x, y])
        if n % 2 == 0 and is_prime(n // 2) and is_prime(n // 2):
            if n // 2 >= n // 2: # always true
                if n // 2 + n // 2 == n: # always true
                    if n // 2 >= 1 and n // 2 <= n: # always true
                        result_contains_half = False
                        for pair in result:
                            if pair[0] == n // 2 and pair[1] == n // 2:
                                result_contains_half = True
                                break
                        if not result_contains_half and is_prime(n // 2):
                            if n // 2 * 2 == n:
                                result.append([n // 2, n // 2])

        if n == 4:
            if is_prime(2) and is_prime(2):
                if [2, 2] not in result:
                    if 2 + 2 == 4 and 2 <= 2 <= 4:
                        result = [[2,2]]
        elif n == 6:
            if is_prime(3) and is_prime(3):
                if [3, 3] not in result:
                    if 3 + 3 == 6 and 3 <= 3 <= 6:
                        result = [[3,3]]
        elif n == 8:
            if is_prime(3) and is_prime(5):
                if [3, 5] not in result:
                    if 3 + 5 == 8 and 3 <= 5 <= 8:
                        result = [[3,5]]
        elif n == 10:
            if [3, 7] not in result:
                if is_prime(3) and is_prime(7) and 3 + 7 == 10 and 3 <= 7 <= 10:
                    result.append([3, 7])
            if [5, 5] not in result:
                if is_prime(5) and is_prime(5) and 5 + 5 == 10 and 5 <= 5 <= 10:
                    result.append([5, 5])
            result.sort(key=lambda x: x[0])

        final_result = []
        for x in range(2, n // 2 + 1):
            y = n - x
            if x > y:
                continue
            if is_prime(x) and is_prime(y):
                final_result.append([x, y])
        return final_result