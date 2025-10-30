import sys
sys.setrecursionlimit(1000000)

def inv_count(arr):
    def merge_sort_indices(indices):
        if len(indices) <= 1:
            return indices, 0
        mid = len(indices) // 2
        left, inv_left = merge_sort_indices(indices[:mid])
        right, inv_right = merge_sort_indices(indices[mid:])
        merged, inv_merge = merge(left, right)
        total_inv = inv_left + inv_right + inv_merge
        return merged, total_inv
    
    def merge(left, right):
        i, j = 0, 0
        inv_count_merge = 0
        merged = []
        while i < len(left) and j < len(right):
            if arr[left[i]] <= arr[right[j]]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inv_count_merge += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv_count_merge
    
    indices = list(range(len(arr)))
    _, inv = merge_sort_indices(indices)
    return inv

data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))

# Compute count and sum of positions for each value
count_v = [0] * M
sum_pos = [0] * M
for i in range(N):
    val = A[i]
    pos = i + 1  # 1-based position
    count_v[val] += 1
    sum_pos[val] += pos

# Compute initial inversion count for k=0
inv_current = inv_count(A)
print(inv_current)  # Output for k=0

# Now for k from 1 to M-1, compute and output using deltas
for k in range(0, M - 1):  # k from 0 to M-2
    v = (M - 1) - k  # v_k
    delta = 2 * sum_pos[v] - count_v[v] * (N + 1)
    inv_current += delta
    print(inv_current)  # Output for k+1