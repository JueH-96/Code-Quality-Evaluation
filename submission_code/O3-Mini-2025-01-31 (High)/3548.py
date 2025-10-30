import math
import itertools

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # Precompute factorials up to n.
        fact = [math.factorial(i) for i in range(n+1)]
        
        # Generator for frequency distributions (multisets) for digits 0..9 summing to n.
        def gen_freq(index, remaining, current):
            if index == 10:
                if remaining == 0:
                    yield current.copy()
                return
            for num in range(remaining + 1):
                current[index] = num
                yield from gen_freq(index+1, remaining-num, current)
        
        # Given a distribution m (a list of 10 counts summing to n),
        # count the number of n-digit integers (i.e. arrangements with no leading zero).
        def count_valid_permutations(m):
            total = fact[n]
            for cnt in m:
                total //= math.factorial(cnt)
            invalid = 0
            if m[0] > 0:
                # Count arrangements where the first digit is 0.
                prod = 1
                for d in range(1, 10):
                    prod *= math.factorial(m[d])
                invalid = fact[n-1] // (math.factorial(m[0]-1) * prod)
            return total - invalid
        
        # Given the frequency distribution m (digits used in the integer),
        # check whether it is possible to rearrange its digits into a palindrome that
        # (a) does not start with zero and 
        # (b) is divisible by k.
        def valid_palindrome_exists(m):
            # First, a necessary condition for palindromic rearrangement:
            if n % 2 == 0:
                # Even length: all counts must be even.
                for cnt in m:
                    if cnt % 2 != 0:
                        return False
            else:
                # Odd length: exactly one count is odd.
                odd_count = sum(cnt % 2 for cnt in m)
                if odd_count != 1:
                    return False
            
            # Special case: for n == 1, the number is simply that single digit.
            if n == 1:
                for d in range(1, 10):
                    if m[d] == 1:
                        return (d % k == 0)
                return False  # Should not happen.
            
            # For n >= 2, we now try to “build” a palindrome.
            L = n // 2  # length of the left half.
            # In any palindromic arrangement, the left half gets exactly m[d]//2 copies of digit d.
            half_counts = [m[d] // 2 for d in range(10)]
            # In a valid palindrome the most-significant digit (first digit of the left half)
            # must be nonzero. So the left half multiset must contain some nonzero digit.
            if L > 0 and sum(half_counts[d] for d in range(1, 10)) == 0:
                return False
            
            # Construct the list for the left half (order not fixed yet).
            left_list = []
            for d in range(10):
                left_list.extend([d] * half_counts[d])
            
            # For odd n, determine the unique middle digit.
            mid_digit = None
            if n % 2 == 1:
                for d in range(10):
                    if m[d] % 2 == 1:
                        mid_digit = d
                        break
            
            # Try every distinct permutation of the left half.
            seen = set()
            for perm in itertools.permutations(left_list):
                if perm in seen:
                    continue
                seen.add(perm)
                # Enforce that the left half does not start with 0.
                if perm and perm[0] == 0:
                    continue
                # Build the full palindrome.
                if n % 2 == 0:
                    full_tuple = perm + perm[::-1]
                else:
                    full_tuple = perm + (mid_digit,) + perm[::-1]
                # Compute the integer from the digits.
                num = 0
                for d in full_tuple:
                    num = num * 10 + d
                if num % k == 0:
                    return True
            return False
        
        total_ans = 0
        freq = [0] * 10
        # Enumerate every multiset m of n digits.
        for m in gen_freq(0, n, freq):
            # Skip the all-zero case (e.g. m[0]==n) because an n-digit number may not be "0...0"
            if m[0] == n:
                continue
            # Only if the digits (as a multiset) can be rearranged into a valid k-palindromic
            # integer do we add the number of n-digit numbers with that multiset.
            if valid_palindrome_exists(m):
                total_ans += count_valid_permutations(m)
        return total_ans


# For testing:
if __name__ == '__main__':
    sol = Solution()
    print(sol.countGoodIntegers(3, 5))  # Expected output: 27
    print(sol.countGoodIntegers(1, 4))  # Expected output: 2
    print(sol.countGoodIntegers(5, 6))  # Expected output: 2468