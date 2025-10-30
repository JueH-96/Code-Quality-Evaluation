from collections import defaultdict

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Create a dictionary to store the indices of each number
    indices = defaultdict(list)
    for i, a in enumerate(A):
        indices[a].append(i)

    # Calculate f(L, R) for each pair (L, R)
    total = 0
    for L in range(1, N+1):
        for R in range(L, N+1):
            # Find the minimum number of operations needed to erase all integers from L to R
            ops = 0
            start = L - 1
            while start < R:
                # Find the range of integers that appear at least once on the blackboard
                min_val = float('inf')
                max_val = -1
                for i in range(start, R):
                    min_val = min(min_val, A[i])
                    max_val = max(max_val, A[i])
                    if max_val - min_val == i - start:
                        # All integers from min_val to max_val appear on the blackboard
                        ops += 1
                        start = i + 1
                        break
                else:
                    # No such range found, so the blackboard cannot be cleared
                    ops = float('inf')
                    break
            total += ops
    return total

print(solve())