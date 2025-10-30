def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])

    max_fact = N * K
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i

    # Compute S
    numerator = fact[N * K]
    denominator = (fact[K] ** N)
    S = numerator // denominator
    m = (S + 1) // 2

    current_sequence = []
    current_counts = [K] * N
    current_product = 1
    for i in range(N):
        current_product *= fact[current_counts[i]]

    remaining_length = N * K

    while len(current_sequence) < remaining_length:
        for x in range(1, N + 1):
            idx = x - 1
            if current_counts[idx] == 0:
                continue

            # Calculate new_product
            new_product = current_product // current_counts[idx]

            # Calculate the number of sequences
            if remaining_length - 1 >= 0:
                numerator_val = fact[remaining_length - 1]
            else:
                numerator_val = 1
            num_sequences = numerator_val // new_product

            if num_sequences < m:
                m -= num_sequences
                continue
            else:
                # Choose this x
                current_sequence.append(x)
                current_counts[idx] -= 1
                current_product = new_product
                remaining_length -= 1
                break
        else:
            # This should not happen as per problem constraints
            pass

    print(' '.join(map(str, current_sequence)))

if __name__ == '__main__':
    main()