import sys

def main() -> None:
    MOD = 10 ** 8                     # 10^8
    
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N = data[0]
    A = data[1:]
    
    # basic statistics
    total_sum = sum(A)                # Σ A_i
    total_pairs = N * (N - 1) // 2    # number of i < j pairs
    
    # count pairs whose sum is smaller than MOD
    A.sort()
    i, j = 0, N - 1
    cnt_lt = 0                        # pairs with A_i + A_j < MOD
    while i < j:
        if A[i] + A[j] < MOD:
            cnt_lt += j - i           # every index in [i+1, j] with i forms a valid pair
            i += 1
        else:
            j -= 1
    
    pairs_ge = total_pairs - cnt_lt   # pairs with A_i + A_j ≥ MOD
    
    # Σ (A_i + A_j) for i < j  = (N-1) * Σ A_i
    raw_sum = (N - 1) * total_sum
    answer = raw_sum - MOD * pairs_ge
    
    print(answer)

if __name__ == "__main__":
    main()