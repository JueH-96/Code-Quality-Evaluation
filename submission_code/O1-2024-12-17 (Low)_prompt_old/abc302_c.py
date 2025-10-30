def solve():
    import sys
    from itertools import permutations

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    strings = data[2:]

    # Helper function to check if two strings differ by exactly 1 character
    def differ_by_one(a, b):
        diff_count = 0
        for x, y in zip(a, b):
            if x != y:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1

    # Generate all permutations of the given strings
    # and check if any one of them satisfies the condition.
    for perm in permutations(strings):
        valid = True
        for i in range(N - 1):
            if not differ_by_one(perm[i], perm[i + 1]):
                valid = False
                break
        if valid:
            print("Yes")
            return

    print("No")

# Call solve() after defining it
solve()