import sys
import itertools

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Special cases:
    # If K == 1 then the answer is simply the maximum element.
    if K == 1:
        sys.stdout.write(str(max(A)))
        return
    # If K == N then the answer is the XOR of all elements.
    if K == N:
        total = 0
        for a in A:
            total ^= a
        sys.stdout.write(str(total))
        return
        
    # In general, we are to choose K distinct elements. 
    # The number of ways to pick K elements is guaranteed to be at most 10^6.
    # To reduce the number of iterations, note that because 
    # C(N, K) = C(N, N-K) we can instead work with the smaller of K and N-K.
    # If K > N-K it means that the chosen set is almost all of A.
    # Then if we let omitted subset = S (of size N-K), we have:
    #    XOR(chosen) = XOR(all) XOR XOR(S).
    # So we iterate over combinations for the omitted subset.
    
    full_xor = 0
    for a in A:
        full_xor ^= a

    # Use the smaller subset size.
    if K <= N - K:
        subset_size = K
        best = 0
        # Iterate over all combinations of A taking K elements.
        for combo in itertools.combinations(A, subset_size):
            xor_val = 0
            for x in combo:
                xor_val ^= x
            if xor_val > best:
                best = xor_val
        sys.stdout.write(str(best))
    else:
        subset_size = N - K  # choose the omitted elements set
        best = 0
        # For each omitted subset, compute the XOR of the chosen elements as:
        # chosen_xor = full_xor XOR (XOR of omitted subset)
        for combo in itertools.combinations(A, subset_size):
            omitted_xor = 0
            for x in combo:
                omitted_xor ^= x
            candidate = full_xor ^ omitted_xor
            if candidate > best:
                best = candidate
        sys.stdout.write(str(best))

if __name__ == '__main__':
    main()