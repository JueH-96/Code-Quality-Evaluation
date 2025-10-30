# YOUR CODE HERE
import sys
import threading

def main():
    import random
    N, X = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    val_to_indices = {}
    for idx, val in enumerate(A):
        if val in val_to_indices:
            val_to_indices[val].append(idx)
        else:
            val_to_indices[val] = [idx]

    # Try random sampling for larger N
    T = 500000  # Number of trials
    for _ in range(T):
        i, j = random.sample(range(N), 2)
        target = X - A[i] - A[j]
        if target in val_to_indices:
            for k in val_to_indices[target]:
                if k != i and k != j:
                    indices = [i, j, k]
                    indices.sort()
                    print(indices[0]+1, indices[1]+1, indices[2]+1)
                    return

    # Try O(N^2) solution for smaller N
    if N <= 5000:
        for i in range(N):
            for j in range(i+1, N):
                target = X - A[i] - A[j]
                if target in val_to_indices:
                    for k in val_to_indices[target]:
                        if k != i and k != j:
                            indices = [i, j, k]
                            indices.sort()
                            print(indices[0]+1, indices[1]+1, indices[2]+1)
                            return
        print(-1)
        return

    # Try fixed number of iterations of two pointers approach
    A_with_idx = [(A[i], i) for i in range(N)]
    A_with_idx.sort()
    MAX_ITER = 1000
    for iter_num in range(min(N, MAX_ITER)):
        i = random.randint(0, N-1)
        ai, idx_i = A_with_idx[i]
        target = X - ai
        left, right = 0, N - 1
        while left < right:
            if left == i:
                left += 1
                continue
            if right == i:
                right -=1
                continue
            aj, idx_j = A_with_idx[left]
            ak, idx_k = A_with_idx[right]
            curr_sum = aj + ak
            if curr_sum == target:
                indices = [idx_i, idx_j, idx_k]
                if len(set(indices)) < 3:
                    if left+1 < right:
                        left += 1
                    else:
                        break
                    continue
                indices.sort()
                print(indices[0]+1, indices[1]+1, indices[2]+1)
                return
            elif curr_sum < target:
                left +=1
            else:
                right -=1
    print(-1)

threading.Thread(target=main).start()