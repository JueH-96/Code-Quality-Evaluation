def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MAXA = 10**5

    # Build smallest-prime-factor (spf) sieve
    spf = list(range(MAXA+1))  # spf[x] = smallest prime factor of x
    for i in range(2, int(MAXA**0.5)+1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAXA+1, i):
                if spf[j] == j:
                    spf[j] = i

    # Function to return sum of prime exponents of x
    def prime_exp_sum(x):
        count = 0
        while x > 1:
            count += 1
            x //= spf[x]
        return count

    # Compute the nim-sum of Grundy numbers
    nim_sum = 0
    for val in A:
        nim_sum ^= prime_exp_sum(val)

    # If nim-sum != 0, first player (Anna) wins; else Bruno wins
    if nim_sum != 0:
        print("Anna")
    else:
        print("Bruno")

# Do NOT forget to call main()
main()