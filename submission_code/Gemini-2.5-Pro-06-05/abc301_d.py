# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input string S and integer N, then computes and prints the result.
    """
    try:
        S = sys.stdin.readline().strip()
        N = int(sys.stdin.readline().strip())
    except (IOError, ValueError):
        # In case of malformed input, though constraints likely prevent this.
        return

    L = len(S)

    # Calculate the number by treating all '?' as '0'.
    # This gives us the smallest possible number that can be formed from S.
    ans = int(S.replace('?', '0'), 2)

    # If the smallest possible number is already larger than N,
    # then no number formed from S can be less than or equal to N.
    if ans > N:
        print(-1)
        return

    # Now, we have a valid candidate 'ans'. We try to make it as large as
    # possible without exceeding N. We do this by greedily changing '?'
    # from '0' to '1', starting from the most significant bit.
    for i in range(L):
        # We only consider positions that were originally '?'.
        if S[i] == '?':
            # The value this bit adds if it is set to '1'.
            # The bit position from the right (0-indexed) is (L - 1 - i).
            # The value is 2^(L - 1 - i).
            bit_value = 1 << (L - 1 - i)
            
            # If we can flip this bit to '1' without exceeding N, we do it.
            # This is an optimal greedy choice because a '1' at a more
            # significant bit position adds more value than any combination
            # of '1's at less significant positions.
            if ans + bit_value <= N:
                ans += bit_value
    
    print(ans)

solve()