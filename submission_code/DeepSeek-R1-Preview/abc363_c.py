import itertools

def main():
    n, k = map(int, input().split())
    s = input().strip()

    # Generate all unique permutations
    unique_perms = set()
    for p in itertools.permutations(s):
        unique_perms.add(''.join(p))
    
    count = 0

    # Check each unique permutation
    for perm in unique_perms:
        is_valid = True
        # Check all possible substrings of length K
        for i in range(len(perm) - k + 1):
            substr = perm[i:i+k]
            if substr == substr[::-1]:
                is_valid = False
                break
        if is_valid:
            count += 1

    print(count)

if __name__ == "__main__":
    main()