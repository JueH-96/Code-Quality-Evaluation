def can_make_identical(N, K, A, B):
    # Check if the frequency of each element is the same in A and B
    from collections import defaultdict
    freq_A = defaultdict(int)
    freq_B = defaultdict(int)
    for a in A:
        freq_A[a] += 1
    for b in B:
        freq_B[b] += 1
    if freq_A != freq_B:
        return False
    # Now, check if for each position, the required element is within K distance
    # We need to ensure that for each i, B[i] is in the multiset of A[j] where |i-j| <= K
    # To do this, we can create a sliding window of size 2K+1 and check if B[i] is in the current window
    # However, a more efficient way is to precompute the positions of each element in A
    # and then for each B[i], check if there is an A[j] with |i-j| <= K and A[j] == B[i]
    # To optimize, we can create a list of lists where each element is mapped to its positions in A
    pos = defaultdict(list)
    for idx, a in enumerate(A):
        pos[a].append(idx)
    # Now, for each B[i], we need to find if there is a j in pos[B[i]] such that |i-j| <= K
    # We can use binary search to find the closest j to i in pos[B[i]]
    from bisect import bisect_left, bisect_right
    for i in range(N):
        b = B[i]
        if b not in pos:
            return False
        # Find the indices in pos[b] that are within [i-K, i+K]
        low = i - K
        high = i + K
        # Find the leftmost index >= low
        left = bisect_left(pos[b], low)
        # Find the rightmost index <= high
        right = bisect_right(pos[b], high)
        if left >= right:
            return False
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        K = int(data[idx+1])
        idx += 2
        A = list(map(int, data[idx:idx+N]))
        idx += N
        B = list(map(int, data[idx:idx+N]))
        idx += N
        if can_make_identical(N, K, A, B):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()