def solve():
    n, m = map(int, input().split())
    mod = 998244353

    primes = []
    temp_m = m
    i = 2
    while i * i <= temp_m:
        if temp_m % i == 0:
            primes.append(i)
            while temp_m % i == 0:
                temp_m //= i
        i += 1
    if temp_m > 1:
        primes.append(temp_m)

    def get_prime_exponents(num):
        exponents = {}
        temp_num = num
        for p in primes:
            count = 0
            while temp_num % p == 0 and p > 0:
                count += 1
                temp_num //= p
            if p > 0:
                exponents[p] = count
        return exponents

    sum_of_scores = 0
    for length in range(1, int(n) + 1):
        def generate_sequences(current_length, current_sequence):
            nonlocal sum_of_scores
            if current_length == length:
                product = 1
                for x in current_sequence:
                    product *= x

                def count_divisors(num):
                    count = 1
                    i = 2
                    temp_num = num
                    while i * i <= temp_num:
                        if temp_num % i == 0:
                            exponent = 0
                            while temp_num % i == 0:
                                exponent += 1
                                temp_num //= i
                            count *= (exponent + 1)
                        i += 1
                    if temp_num > 1:
                        count *= 2
                    return count

                score = 1
                temp_product = product
                i = 2
                while i * i <= temp_product:
                    if temp_product % i == 0:
                        exponent = 0
                        while temp_product % i == 0:
                            exponent += 1
                            temp_product //= i
                        score = (score * (exponent + 1)) % mod
                    i += 1
                if temp_product > 1:
                    score = (score * 2) % mod

                sum_of_scores = (sum_of_scores + score) % mod
                return

            for i in range(1, m + 1):
                generate_sequences(current_length + 1, current_sequence + [i])

        generate_sequences(0, [])

    print(sum_of_scores)

solve()