import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    K = int(input[idx+1])
    idx += 2
    A = list(map(int, input[idx:idx+N]))
    idx += N

    # Binary search for x
    left = min(A)
    right = sum(A)
    best = left

    def is_possible(x):
        total = sum(A)
        if total < K * x:
            return False
        current_sum = 0
        splits = 0
        max_start = 0
        n = N
        for i in range(2 * n):
            current_sum += A[i % n]
            while current_sum >= x and splits < K:
                current_sum -= A[(max_start) % n]
                splits += 1
                max_start += 1
                if splits == K:
                    return True
        return False

    best = left
    while left <= right:
        mid = (left + right) // 2
        if is_possible(mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    x = best

    # Find all intervals with sum x
    doubled = A * 2
    n = N
    prefix = [0] * (2 * n + 1)
    for i in range(2 * n):
        prefix[i + 1] = prefix[i] + doubled[i]

    sum_map = defaultdict(list)
    sum_map[0].append(-1)
    cut_lines = set()

    for i in range(2 * n):
        current_sum = prefix[i + 1]
        target = current_sum - x
        if target in sum_map:
            for j in sum_map[target]:
                start = j + 1
                end = i
                length = end - start + 1
                if length > n:
                    continue
                original_start = start % n
                original_end = end % n
                left_cut = (original_start - 1) % n
                right_cut = original_end % n
                cut_lines.add(left_cut)
                cut_lines.add(right_cut)
        sum_map[current_sum].append(i)

    y = n - len(cut_lines)
    print(x, y)

if __name__ == '__main__':
    main()