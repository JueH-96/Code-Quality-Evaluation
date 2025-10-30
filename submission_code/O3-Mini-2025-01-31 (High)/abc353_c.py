def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    A = list(map(int, data[1:]))
    mod_val = 10**8

    # Compute sum of all A_i.
    total_A = sum(A)
    # Notice that: sum_{i<j} (A_i + A_j) = (N-1)*sum(A)
    total_pair_sum = (N - 1) * total_A

    # We need to subtract mod_val for every pair (i, j) (i<j)
    # such that A_i + A_j >= mod_val. Because:
    #   f(A_i, A_j) = (A_i + A_j) % mod_val
    #              = (A_i + A_j) - mod_val   if A_i+A_j >= mod_val
    #              = (A_i + A_j)           otherwise.
    # Thus, the answer = total_pair_sum - mod_val*(# of pairs with A_i+A_j>=mod_val).

    # Count the pairs with A_i + A_j >= mod_val using two pointers after sorting.
    A.sort()
    count = 0
    i = 0
    j = N - 1
    while i < j:
        if A[i] + A[j] >= mod_val:
            # All indices from i to j-1 when paired with A[j] will make a valid pair.
            count += (j - i)
            j -= 1
        else:
            i += 1

    ans = total_pair_sum - count * mod_val
    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()