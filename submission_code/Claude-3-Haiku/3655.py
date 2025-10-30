from typing import Set

class Solution:
    def is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    def minOperations(self, n: int, m: int) -> int:
        if self.is_prime(n) or self.is_prime(m):
            return -1

        n_digits = list(str(n))
        m_digits = list(str(m))
        cost = 0

        for i in range(len(n_digits)):
            if n_digits[i] != m_digits[i]:
                if n_digits[i] < m_digits[i]:
                    while n_digits[i] != m_digits[i]:
                        if n_digits[i] == '9':
                            break
                        n_digits[i] = str(int(n_digits[i]) + 1)
                        cost += int(n_digits[i])
                else:
                    while n_digits[i] != m_digits[i]:
                        if n_digits[i] == '0':
                            break
                        n_digits[i] = str(int(n_digits[i]) - 1)
                        cost += int(n_digits[i])

        return cost