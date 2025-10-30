def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    max_A = max(A)
    
    # Initialize array to hold the count of prime factors
    prime_factors_count = [0] * (max_A + 1)
    
    # Sieve of Eratosthenes to count prime factors
    for p in range(2, max_A + 1):
        if prime_factors_count[p] == 0:
            # p is a prime
            for multiple in range(p, max_A + 1, p):
                while multiple % p == 0:
                    prime_factors_count[multiple] += 1
                    multiple //= p
    
    # Calculate the XOR of all Grundy numbers
    xor = 0
    for num in A:
        xor ^= prime_factors_count[num]
    
    # Determine the winner
    if xor != 0:
        print("Anna")
    else:
        print("Bruno")

if __name__ == "__main__":
    main()