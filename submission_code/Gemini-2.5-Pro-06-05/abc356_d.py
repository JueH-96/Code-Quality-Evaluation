import sys

def solve():
    """
    Reads input N and M, calculates the specified sum, and prints the result.
    The calculation is done modulo 998244353.
    """
    try:
        # Read N and M from a single line of stdin.
        N, M = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Gracefully exit on empty input or parsing errors.
        return

    MOD = 998244353

    ans = 0
    # The sum is over k from 0 to N, which includes N + 1 numbers.
    num_elements = N + 1

    # N and M are less than 2^60, so we only need to consider bits up to 59.
    # Looping up to 60 (i.e., range(61)) is safe and covers all possibilities.
    for i in range(61):
        # We only consider bit positions 'i' where the i-th bit of M is set.
        if not ((M >> i) & 1):
            continue

        # If the i-th bit of M is 1, we count how many numbers k in [0, N]
        # have their i-th bit set. This count contributes to the total sum.
        
        # The i-th bit pattern repeats every 2^(i+1).
        # A full period consists of 2^i zeros followed by 2^i ones.
        p = 1 << i           # 2^i
        period = p * 2       # 2^(i+1)

        # Calculate the number of full periods within the 'num_elements' range.
        num_periods = num_elements // period
        # Each full period has 'p' numbers with the i-th bit set.
        count = num_periods * p

        # Calculate the contribution from the remaining elements.
        rem = num_elements % period
        # In the remainder range [0, rem-1], numbers from 'p' onwards have the i-th bit set.
        count += max(0, rem - p)

        # Add the count for this bit to the total answer, taking modulo.
        ans = (ans + count) % MOD

    print(ans)

if __name__ == "__main__":
    solve()