def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MAX_A = 10**5

    # Precompute smallest prime factors (spf) up to MAX_A using a sieve
    spf = [0] * (MAX_A + 1)
    spf[1] = 1
    for i in range(2, MAX_A + 1):
        if spf[i] == 0:  # i is prime
            spf[i] = i
            for j in range(i * i, MAX_A + 1, i):
                if spf[j] == 0:
                    spf[j] = i

    # Precompute the "sum of prime exponents" for each number up to MAX_A.
    # sum_exp[x] = total number of prime factors counting multiplicities
    sum_exp = [0] * (MAX_A + 1)
    sum_exp[1] = 0
    for x in range(2, MAX_A + 1):
        # Let p = spf[x], then x = p * (x // p)
        # sum_exp[x] = sum_exp[x // p] + 1
        p = spf[x]
        sum_exp[x] = sum_exp[x // p] + 1

    # Compute the nim-sum (xor) of sum_exp[A_i].
    nim_sum = 0
    for val in A:
        nim_sum ^= sum_exp[val]

    # If nim-sum != 0, Anna wins, else Bruno
    if nim_sum != 0:
        print("Anna")
    else:
        print("Bruno")

def main():
    solve()

# Call solve() if needed
if __name__ == "__main__":
    main()