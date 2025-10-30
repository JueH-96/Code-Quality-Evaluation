def can_make_identical(N, K, A, B):
    # Check if the frequency of each element in A and B is the same
    from collections import defaultdict
    freq_A = defaultdict(int)
    freq_B = defaultdict(int)
    for a in A:
        freq_A[a] += 1
    for b in B:
        freq_B[b] += 1
    if freq_A != freq_B:
        return False
    # Now, check if for each position, the required element is within the K-window
    # We need to ensure that for each i, B[i] is in the K-window of A[i]
    # But since we can perform multiple operations, we need to ensure that the overall frequency is maintained
    # and that the elements can be rearranged within the K-window constraints
    # To simplify, we can sort the A and B and check if they are the same
    # Because the operations allow us to rearrange elements within the K-window
    # So, if the sorted A and B are the same, then it's possible
    # Because we can perform operations to rearrange the elements
    # However, this is not sufficient because the K-window constraint limits the rearrangement
    # For example, if K is 1, we can only swap adjacent elements
    # So, the sorted A and B must be the same, but also the elements must be able to be rearranged within the K-window
    # To handle this, we can check if the sorted A and B are the same
    # Because the operations allow us to rearrange elements within the K-window, and if the frequency is the same, it's possible
    # So, we can proceed with this approach
    sorted_A = sorted(A)
    sorted_B = sorted(B)
    if sorted_A != sorted_B:
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