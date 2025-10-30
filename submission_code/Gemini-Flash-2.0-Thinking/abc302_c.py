import itertools

def solve():
    n, m = map(int, input().split())
    strings = [input() for _ in range(n)]

    def differs_by_one(s1, s2):
        diff_count = 0
        for i in range(m):
            if s1[i] != s2[i]:
                diff_count += 1
        return diff_count == 1

    for perm in itertools.permutations(strings):
        is_valid = True
        for i in range(n - 1):
            if not differs_by_one(perm[i], perm[i+1]):
                is_valid = False
                break
        if is_valid:
            print("Yes")
            return

    print("No")

# YOUR CODE HERE
solve()