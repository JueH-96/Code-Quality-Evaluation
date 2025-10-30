import sys
import bisect

def compute_prefix_k(S, T):
    k = 0
    for char in S:
        if k < len(T) and char == T[k]:
            k += 1
    return k

def compute_suffix_k(S, T):
    rev_S = S[::-1]
    rev_T = T[::-1]
    prefix_k_reversed = compute_prefix_k(rev_S, rev_T)
    suffix_k = len(T) - prefix_k_reversed
    return suffix_k

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    T = input[1]
    S = input[2:2+N]
    
    m = len(T)
    prefix_k = [compute_prefix_k(s, T) for s in S]
    suffix_k = [compute_suffix_k(s, T) for s in S]
    
    suffix_k_sorted = sorted(suffix_k)
    
    total = 0
    for k in prefix_k:
        if k >= m:
            total += N
        else:
            remaining = m - k
            idx = bisect.bisect_left(suffix_k_sorted, remaining)
            total += N - idx
    print(total)

if __name__ == "__main__":
    main()