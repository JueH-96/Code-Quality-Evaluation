from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        """
        Finds the k-th lexicographically sorted alternating permutation of numbers from 1 to n.
        """
        
        # Precompute factorials up to n. Python's arbitrary-precision integers handle large values.
        fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = fact[i - 1] * i

        # Count the number of odd and even integers in the range [1, n].
        num_odd = (n + 1) // 2
        num_even = n // 2

        # Calculate the total number of valid alternating permutations.
        total_perms = 0
        if num_odd == num_even:
            # Two types of patterns: odd-even-... and even-odd-...
            # Each has fact[num_odd] * fact[num_even] permutations.
            total_perms = 2 * fact[num_odd] * fact[num_even]
        else: # num_odd == num_even + 1 (since n is odd)
            # Must start with odd: odd-even-odd-...
            total_perms = fact[num_odd] * fact[num_even]

        if k > total_perms:
            return []

        # Convert k from 1-based to 0-based index for calculations.
        k -= 1
        
        result = []
        used = [False] * (n + 1)
        
        # Keep track of the number of remaining odd and even numbers.
        o, e = num_odd, num_even

        # Keep track of the parity of the last placed number.
        # None: initial state, True: last was odd, False: last was even.
        last_is_odd = None

        for _ in range(n):
            if last_is_odd is True: # The last number placed was odd, so the current one must be even.
                for num in range(2, n + 1, 2):
                    if not used[num]:
                        count = fact[o] * fact[e - 1]
                        if k < count:
                            result.append(num)
                            used[num] = True
                            e -= 1
                            last_is_odd = False
                            break
                        else:
                            k -= count
            elif last_is_odd is False: # The last number placed was even, so the current one must be odd.
                for num in range(1, n + 1, 2):
                    if not used[num]:
                        count = fact[e] * fact[o - 1]
                        if k < count:
                            result.append(num)
                            used[num] = True
                            o -= 1
                            last_is_odd = True
                            break
                        else:
                            k -= count
            else: # This is the first number to be placed.
                # We iterate through all numbers from 1 to n to maintain lexicographical order.
                for num in range(1, n + 1):
                    if num % 2 == 1: # Current number to test is odd.
                        if e >= o - 1:
                            count = fact[e] * fact[o - 1]
                            if k < count:
                                result.append(num)
                                used[num] = True
                                o -= 1
                                last_is_odd = True
                                break
                            else:
                                k -= count
                    else: # Current number to test is even.
                        if o >= e - 1:
                            count = fact[o] * fact[e - 1]
                            if k < count:
                                result.append(num)
                                used[num] = True
                                e -= 1
                                last_is_odd = False
                                break
                            else:
                                k -= count
        
        return result