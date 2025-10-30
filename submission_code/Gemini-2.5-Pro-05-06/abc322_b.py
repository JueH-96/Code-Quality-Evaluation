# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()

    # Check if S is a prefix of T
    # S is a prefix of T if the first N characters of T coincide with S.
    # Since len(S) is N, T.startswith(S) correctly checks this.
    is_prefix = t.startswith(s)

    # Check if S is a suffix of T
    # S is a suffix of T if the last N characters of T coincide with S.
    # Since len(S) is N, T.endswith(S) correctly checks this.
    is_suffix = t.endswith(s)

    if is_prefix and is_suffix:
        print(0)
    elif is_prefix:  # and not is_suffix (implicitly)
        print(1)
    elif is_suffix:  # and not is_prefix (implicitly)
        print(2)
    else:  # not is_prefix and not is_suffix
        print(3)

if __name__ == '__main__':
    solve()