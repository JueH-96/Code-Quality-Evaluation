import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    k = int(input[1])
    s = input[2]

    unique_perms = set()

    # Generate all unique permutations
    for p in itertools.permutations(s):
        unique_perms.add(''.join(p))
    
    # Function to check if any palindrome substring of length k exists
    def has_palindrome(subs, k):
        for i in range(len(subs) - k + 1):
            candidate = subs[i:i+k]
            if candidate == candidate[::-1]:
                return True
        return False
    
    count = 0
    for perm in unique_perms:
        if not has_palindrome(perm, k):
            count += 1
    
    print(count)

if __name__ == "__main__":
    main()