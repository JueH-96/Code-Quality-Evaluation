def solve():
    import sys
    input_data = sys.stdin.read().strip()
    N = int(input_data)

    # We want the largest integer c <= N such that c = x^3 for some x
    # and c in decimal form is a palindrome.

    # A direct approach: 
    #   1. Find the integer cube root boundary ~ floor(N^(1/3)).
    #   2. For x from 1 to that boundary+2, compute x^3.
    #   3. If x^3 <= N and is a palindrome, keep track of the maximum.
    # This approach is feasible because N <= 10^18 => x <= 10^6.

    def is_palindrome(num):
        s = str(num)
        return s == s[::-1]

    # Find an integer cube root bound (by binary searching or approximate).
    # We'll use an approximate approach, then adjust if needed.
    approx_cbrt = int(N ** (1/3))
    # Adjust to ensure we have a correct lower or upper bound.
    # We'll move up or down until we find the largest x such that x^3 <= N.
    while (approx_cbrt + 1) ** 3 <= N:
        approx_cbrt += 1
    while approx_cbrt ** 3 > N:
        approx_cbrt -= 1

    limit = approx_cbrt + 2  # a small safety margin

    best = 0
    for x in range(1, limit + 1):
        c = x**3
        if c > N:
            break
        if is_palindrome(c):
            if c > best:
                best = c

    print(best)

def main():
    solve()

if __name__ == "__main__":
    main()