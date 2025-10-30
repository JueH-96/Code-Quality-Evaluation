def main():
    import sys
    import math

    N, K = map(int, sys.stdin.readline().split())
    total_length = N * K

    # Precompute factorials up to the maximum needed
    max_fact = total_length
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i

    # Compute S = (N*K)! / (K!^N)
    numerator = fact[total_length]
    denominator = fact[K] ** N
    S = numerator // denominator

    target = (S + 1) // 2 - 1  # 0-based index

    remaining = [K] * N
    current_multinom = S
    total_remaining = total_length
    result = []

    for _ in range(total_length):
        for x in range(1, N+1):
            idx = x - 1
            if remaining[idx] == 0:
                continue
            original_count = remaining[idx]
            # Compute the number of sequences starting with x
            cnt = current_multinom * original_count // total_remaining
            if target >= cnt:
                target -= cnt
                continue
            # Choose x
            result.append(x)
            remaining[idx] -= 1
            # Update current_multinom
            current_multinom = current_multinom * original_count // total_remaining
            total_remaining -= 1
            break
        else:
            # This should not happen as remaining has at least one element
            pass

    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()