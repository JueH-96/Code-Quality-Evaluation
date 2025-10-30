def min_groups_linear(A, M):
    count = 1
    total = 0
    for a in A:
        total += a
        if total >= M:
            count += 1
            total = 0
    return count

def can_partition(M, A, K):
    N = len(A)
    # Check linear arrangement
    linear_groups = min_groups_linear(A, M)
    if linear_groups <= K:
        return True
    # Check circular arrangement by combining first and last elements
    combined = A[-1] + A[0]
    new_A = A[1:-1] + [combined]
    circular_groups = min_groups_linear(new_A, M)
    return circular_groups <= K - 1  # Since combined elements count as one group

def find_max_M(A, K):
    total_sum = sum(A)
    max_a = max(A)
    low = max_a
    high = total_sum
    while low < high:
        M = (low + high + 1) // 2
        if can_partition(M, A, K):
            low = M
        else:
            high = M - 1
    return low

def count_never_used_cut_lines(A, M, K):
    N = len(A)
    count = 0
    total = 0
    start = 0
    for i in range(N):
        total += A[i]
        if total < M:
            continue
        else:
            if i - start >= 1:
                count += (i - start - 1)
            start = i + 1
            total = 0
    # Handle circularity
    if start > 0:
        combined = A[start:] + A[:start]
        total = sum(combined)
        if total < M:
            count += (N - start + (start - 1))
    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    M = find_max_M(A, K)
    y = count_never_used_cut_lines(A, M, K)
    print(M, y)

if __name__ == "__main__":
    main()