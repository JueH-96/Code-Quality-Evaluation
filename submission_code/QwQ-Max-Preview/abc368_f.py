import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1])) if N > 0 else []
    
    if N == 0:
        print("Bruno")
        return
    
    max_a = max(A)
    max_spf = 10**5  # Since each A_i is up to 1e5
    
    # Compute smallest prime factors (SPF) using sieve
    spf = list(range(max_spf + 1))
    for i in range(2, int(max_spf**0.5) + 1):
        if spf[i] == i:  # i is a prime
            for j in range(i * i, max_spf + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    xor_total = 0
    for a in A:
        x = a
        sum_exp = 0
        while x > 1:
            p = spf[x]
            while x % p == 0:
                sum_exp += 1
                x //= p
        xor_total ^= sum_exp
    
    print("Anna" if xor_total != 0 else "Bruno")

if __name__ == "__main__":
    main()