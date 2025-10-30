def main():
    import sys
    from itertools import permutations

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    strings = data[2:]
    
    # Function to count character differences
    def diff_count(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    # Try all permutations of the given strings
    for perm in permutations(strings):
        valid = True
        for i in range(N - 1):
            if diff_count(perm[i], perm[i + 1]) != 1:
                valid = False
                break
        if valid:
            print("Yes")
            return

    print("No")

# Do not forget to call main
main()