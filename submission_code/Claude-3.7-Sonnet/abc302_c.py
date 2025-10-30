from itertools import permutations

def differ_by_one(s1, s2):
    """Check if strings s1 and s2 differ by exactly one character."""
    diff_count = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            diff_count += 1
            if diff_count > 1:
                return False
    return diff_count == 1

def main():
    n, m = map(int, input().split())
    strings = [input().strip() for _ in range(n)]
    
    # Check all permutations
    for perm in permutations(strings):
        is_valid = True
        for i in range(n - 1):
            if not differ_by_one(perm[i], perm[i+1]):
                is_valid = False
                break
        
        if is_valid:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()