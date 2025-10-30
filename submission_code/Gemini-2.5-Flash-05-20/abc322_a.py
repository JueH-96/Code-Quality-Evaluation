import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # The problem asks for the smallest integer n (1-indexed) such that
    # S[n-1 : n+2] is "ABC".
    # Python's string.find() method returns the lowest 0-indexed position
    # where the substring is found, or -1 if not found.
    
    found_idx = S.find("ABC")

    if found_idx != -1:
        # If "ABC" is found at 0-indexed position `found_idx`,
        # the 1-indexed position is `found_idx + 1`.
        print(found_idx + 1)
    else:
        # If "ABC" is not found in the string.
        print(-1)

if __name__ == '__main__':
    solve()