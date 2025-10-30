# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Function to count inversions in a sequence
def count_inversions(seq):
    def merge_sort(left, right):
        if left >= right:
            return 0
        mid = (left + right) // 2
        inv_count = merge_sort(left, mid) + merge_sort(mid + 1, right)
        i = j = mid + 1
        for k in range(left, mid + 1):
            while i <= right and seq[k] > seq[i]:
                i += 1
            inv_count += i - (mid + 1)
        seq[left:right + 1] = sorted(seq[left:right + 1])
        return inv_count

    return merge_sort(0, len(seq) - 1)

# Solve the problem
for k in range(M):
    B = [(a + k) % M for a in A]
    print(count_inversions(B))