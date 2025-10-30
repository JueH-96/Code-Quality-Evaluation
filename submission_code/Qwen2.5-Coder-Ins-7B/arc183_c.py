import sys
from math import factorial

MOD = 998244353

def count_valid_permutations(N, M, conditions):
    def count_inversions(arr):
        if len(arr) <= 1:
            return 0
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        return count_inversions(left) + count_inversions(right) + merge_count(left, right)

    def merge_count(left, right):
        i = j = inv_count = 0
        merged = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inv_count += len(left) - i
        merged += left[i:]
        merged += right[j:]
        return inv_count, merged

    def count_valid_permutations_recursive(N, conditions):
        if N == 0:
            return 1
        if N == 1:
            return 1
        count = 0
        for i in range(1, N + 1):
            valid = True
            for L, R, X in conditions:
                if L <= i <= R and i == X:
                    valid = False
                    break
            if valid:
                count += count_valid_permutations_recursive(N - 1, conditions)
        return count

    return count_valid_permutations_recursive(N, conditions) % MOD

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
conditions = []
for i in range(M):
    L = int(data[2 + 3 * i])
    R = int(data[3 + 3 * i])
    X = int(data[4 + 3 * i])
    conditions.append((L, R, X))

# Calculate and print the result
result = count_valid_permutations(N, M, conditions)
print(result)