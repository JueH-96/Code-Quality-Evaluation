def main():
    import sys
    from itertools import permutations

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2:]

    # A helper function to check if two strings differ by exactly one character
    def differ_by_one(str1, str2):
        diff = 0
        for c1, c2 in zip(str1, str2):
            if c1 != c2:
                diff += 1
                if diff > 1:
                    return False
        return diff == 1
    
    # Try all permutations of the strings
    for perm in permutations(S):
        valid = True
        for i in range(N - 1):
            if not differ_by_one(perm[i], perm[i+1]):
                valid = False
                break
        if valid:
            print("Yes")
            return
    print("No")

# Do not forget to call main
main()