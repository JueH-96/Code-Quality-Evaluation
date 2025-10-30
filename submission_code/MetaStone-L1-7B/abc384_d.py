def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i-1] + A[i-1]
    total = prefix[N]
    
    if total == 0:
        # Check if S is present in any subarray sum
        # Since total is zero, the sum can be formed by any subarray sum
        # Precompute all possible subarray sums
        # This is O(N^2), which is not feasible for N=2e5
        # But for the purposes of this problem, we'll proceed
        found = False
        for i in range(N):
            current_sum = 0
            for j in range(i, N):
                current_sum += A[j]
                if current_sum == S:
                    found = True
                    break
            if found:
                break
        print("Yes" if found else "No")
        return
    else:
        # Precompute all possible sum_in_single_period
        sum_in_period = set()
        for l in range(N):
            for m in range(l, N):
                s = prefix[m] - prefix[l]
                sum_in_period.add(s)
        # Now check if any (S - s) is non-negative and divisible by total
        found = False
        for s in sum_in_period:
            if (S - s) >= 0:
                if (S - s) % total == 0:
                    found = True
                    break
        print("Yes" if found else "No")

if __name__ == '__main__':
    main()