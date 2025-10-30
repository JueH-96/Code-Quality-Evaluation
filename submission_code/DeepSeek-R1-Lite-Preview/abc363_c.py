import itertools

def main():
    # Read input
    N_K = input().strip()
    while N_K == '':
        N_K = input().strip()
    N, K = map(int, N_K.split())
    S = input().strip()
    while S == '':
        S = input().strip()
    
    # Generate unique permutations
    perms = set(itertools.permutations(S))
    
    count = 0
    for perm in perms:
        perm_str = ''.join(perm)
        # Check for palindromic substrings of length K
        has_palindrome = False
        for i in range(N - K + 1):
            substr = perm_str[i:i+K]
            if substr == substr[::-1]:
                has_palindrome = True
                break
        if not has_palindrome:
            count += 1
    
    print(count)

if __name__ == '__main__':
    main()