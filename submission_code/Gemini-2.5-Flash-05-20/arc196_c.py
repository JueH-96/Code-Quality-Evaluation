import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    MOD = 998244353

    # Necessary conditions for strong connectivity
    # Vertex 1 must be black to be a target of a backward edge (path from 2N to 1)
    # Vertex 2N must be white to be a source of a backward edge (path from 2N to 1)
    if S[0] == 'W' or S[2 * N - 1] == 'B':
        print(0)
        return

    ans = 1
    # 'balance' represents the number of unmatched 'W' vertices encountered so far.
    # When a 'W' is encountered, balance increases.
    # When a 'B' is encountered, it must be paired with one of the 'balance' available 'W's.
    # If balance is 0 when a 'B' is encountered, it means this 'B' cannot be paired with
    # a preceding 'W' while maintaining the specific structure required for this problem.
    balance = 0

    for i in range(2 * N):
        if S[i] == 'W':
            balance += 1
        else:  # S[i] == 'B'
            ans = (ans * balance) % MOD
            balance -= 1

        # If balance drops below zero, it means we have more 'B's than 'W's
        # that can be paired in a way that contributes to strong connectivity.
        # This breaks the required structure.
        if balance < 0:
            ans = 0
            break

    # After iterating through all characters, the total number of 'W's and 'B's must be equal (N each),
    # so the final balance should be 0. If not, it means the pairing is incomplete or invalid.
    if balance != 0:
        ans = 0

    print(ans)

solve()