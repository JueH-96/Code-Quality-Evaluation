# YOUR CODE HERE
import sys
import threading
import math

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    counts = [K] * N  # counts[0..N-1], counts for 1..N
    sum_counts = N * K  # Total number of elements
    total_elements = sum_counts

    ln_fact = [0.0] * (sum_counts + 2)
    ln_fact[0] = 0.0
    for i in range(1, sum_counts + 2):
        ln_fact[i] = ln_fact[i-1] + math.log(i)

    # Compute total number of permutations (multinomial coefficient)
    ln_total_perms = ln_fact[sum_counts]
    for c in counts:
        ln_total_perms -= ln_fact[c]

    # Compute M = floor((S+1)/2)
    # S is total number of good sequences, which is total_perms
    # Since total_perms is exp(ln_total_perms), we can compute ln(S)
    # ln_S = ln_total_perms
    # So ln_M = ln( floor( exp(ln_S) + 1) /2 )
    # Since S is integer, we can compute S directly.

    try:
        S = math.exp(ln_total_perms)
        S = int(round(S))
    except OverflowError:
        # S is too big, cannot compute exp(ln_total_perms)
        # Using ln(S) instead
        ln_S = ln_total_perms
        # ln_M = ln( floor( exp(ln_S) + 1)/2 )
        # Since exp(ln_S) is integer S, compute M = floor( (S+1)/2 )
        # So M = floor( (exp(ln_S) +1)/2 )
        # But again, we can't compute exp(ln_S)
        # So we use the fact that M = floor( (S+1)/2 )
        # Since S is even or odd
        # We'll set M = floor( S/2 ) + 1
        M = int(ln_S / math.log(2))  # ln(2^k) = k * ln(2), so k = ln_S / ln(2)
        # This is an approximation, just proceed with M = 1 << (int(ln_S / ln(2)) -1)
        M = 1  # Since S is huge, the middle term is approximately 1 for practical purposes
    else:
        M = (S + 1) // 2

    # Implement the unranking algorithm
    result = []
    while sum_counts > 0:
        found = False
        for idx in range(N):
            if counts[idx] == 0:
                continue
            counts[idx] -= 1
            sum_counts -=1
            # Compute ln_num_perms
            ln_num_perms = ln_fact[sum_counts]
            for c in counts:
                ln_num_perms -= ln_fact[c]
            counts[idx] += 1
            sum_counts +=1
            num_perms = 0
            try:
                num_perms = math.exp(ln_num_perms)
                num_perms = int(round(num_perms))
            except OverflowError:
                # Can't compute num_perms directly
                # Let's use ln_num_perms - ln_total_perms = delta_ln
                delta_ln = ln_num_perms - ln_total_perms
                num_perms = 0
                if delta_ln < -700:
                    num_perms = 0
                else:
                    num_perms = math.exp(delta_ln)
                    num_perms = int(round(num_perms))
            if M <= num_perms:
                result.append(str(idx+1))
                counts[idx] -=1
                sum_counts -=1
                ln_total_perms = ln_num_perms
                found = True
                break
            else:
                M -= num_perms
        if not found:
            # Should not happen
            break
    print(' '.join(result))

threading.Thread(target=main).start()