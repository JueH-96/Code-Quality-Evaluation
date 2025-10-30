# YOUR CODE HERE
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    if M == 1:
        print(0)
        return
    
    # The number of ways to assign numbers to the first person is M
    # The number of ways to assign numbers to the second person is M-1 (different from the first person)
    # For the third person to N-th person, each has M-1 choices (different from the previous person)
    # However, we need to ensure that the first and last person are also different
    # So we use the principle of inclusion-exclusion
    
    # Total ways without considering the first and last person being the same
    total_ways = (M * pow(M-1, N-1, MOD)) % MOD
    
    # Ways where the first and last person are the same
    # If the first and last person are the same, we have M choices for the first person
    # And M-1 choices for each of the remaining N-2 people
    same_first_last = (M * pow(M-1, N-2, MOD)) % MOD
    
    # The result is the total ways minus the ways where the first and last person are the same
    result = (total_ways - same_first_last) % MOD
    
    print(result)

if __name__ == "__main__":
    main()