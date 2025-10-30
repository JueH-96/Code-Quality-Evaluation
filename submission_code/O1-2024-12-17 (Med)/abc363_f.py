def main():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline().strip())

    # Quick check: if N = 1, answer is "1".
    if N == 1:
        print("1")
        return

    # A helper to check if a string is a palindrome
    def is_palindrome_str(s: str) -> bool:
        return s == s[::-1]

    # A helper to check if an integer (in base 10) contains no '0'
    def has_no_zero(n: int) -> bool:
        # Faster than converting to string in a tight loop
        while n > 0:
            if n % 10 == 0:
                return False
            n //=10
        return True

    # A helper to reverse the digits of an integer (no leading zeros to worry about, since no digit is zero)
    def reverse_int(n: int) -> int:
        r = 0
        while n > 0:
            r = r*10 + (n%10)
            n //=10
        return r

    # Memo for storing solutions: memo[x] = palindrome-string whose product = x, or None if impossible
    memo = {}

    def search(x: int) -> str or None:
        """
        Returns a palindrome expression (using digits 1..9 and '*') that evaluates to x,
        or None if impossible. Ensures length of the returned string <= 1000 if it exists.
        """
        if x in memo:
            return memo[x]
        # If x itself (as a decimal) has no '0' and is a palindrome, that is a valid one-factor solution
        sx = str(x)
        if '0' not in sx and is_palindrome_str(sx):
            # This is a valid expression
            if len(sx) <= 1000:
                memo[x] = sx
                return sx

        # Otherwise, try to factor x as f * f_rev * (sub-solution)
        # We'll iterate over possible f up to 10^6 because sqrt(10^12)=10^6
        # but break early if f*f_rev > x.
        # We also require f and f_rev have no zero digits.
        limit = 10**6
        # We'll do a standard for-loop from 1..limit, but we'll build a local loop
        # that tries only valid f's (has_no_zero(f)).
        # However, building a big list of all "f" might be too big in memory (up to a million),
        # so we'll just do it on the fly:

        f = 1
        while f <= limit:
            # product f * reverse_int(f)
            # We check quickly if f has zero or not first:
            if has_no_zero(f):
                fr = reverse_int(f)
                # If fr also valid, compute product
                if has_no_zero(fr):
                    prod = f * fr
                    if prod > x:
                        break  # no need to go further in f
                    if x % prod == 0:
                        subx = x // prod
                        subres = search(subx)
                        if subres is not None:
                            f_str = str(f)
                            fr_str = str(fr)
                            # Build the candidate solution by wrapping subres in the middle
                            if len(subres) == 0:
                                # Means subx = 1 gave an empty string, so final is f_str*f_rev
                                candidate = f_str + "*" + fr_str
                            else:
                                candidate = f_str + "*" + subres + "*" + fr_str

                            if len(candidate) <= 1000:
                                memo[x] = candidate
                                return candidate
            f += 1

        # If no construction found, mark as None and return
        memo[x] = None
        return None

    ans = search(N)
    if ans is None or len(ans) > 1000:
        print(-1)
    else:
        print(ans)

# Call main() at the end
if __name__ == "__main__":
    main()