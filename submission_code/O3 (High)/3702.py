from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        # The only possible prime divisors of numbers between 1 and 10
        PRIMES = (2, 3, 5, 7)

        # distinct prime factors (multiplicity is irrelevant for the rule we use)
        def factors(x: int) -> List[int]:
            return [p for p in PRIMES if x % p == 0]

        n = len(nums)
        best = 0

        for left in range(n):
            # how many elements in the current window are divisible by each prime
            cnt = {2: 0, 3: 0, 5: 0, 7: 0}

            for right in range(left, n):
                for p in factors(nums[right]):
                    cnt[p] += 1

                length = right - left + 1

                # decide whether current window is product-equivalent
                if length == 1:
                    valid = nums[left] == 1        # prod = gcd = lcm = 1
                elif length == 2:
                    valid = True                  # always true for two numbers
                else:
                    # For length ≥ 3 each prime may divide at most one element
                    valid = all(v <= 1 for v in cnt.values())

                if valid:
                    best = max(best, length)
                else:
                    # once a length ≥3 window becomes invalid it can never
                    # become valid again by extending further to the right
                    if length >= 3:
                        break

        return best