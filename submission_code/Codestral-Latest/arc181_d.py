import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
P = list(map(int, data[index:index + N]))
index += N
M = int(data[index])
index += 1
A = list(map(int, data[index:index + M]))

# Create a list to store the inversion count after each operation
inversion_counts = []

# Function to count inversions in a list
def count_inversions(arr):
    if len(arr) <= 1:
        return arr, 0
    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])
    merged, split_inv = merge_and_count_split_inversions(left, right)
    return merged, left_inv + right_inv + split_inv

def merge_and_count_split_inversions(left, right):
    sorted_list = []
    i = j = split_inv = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1
            split_inv += len(left) - i
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])
    return sorted_list, split_inv

# Apply operations and count inversions
current_permutation = P.copy()
for k in A:
    for i in range(k - 1):
        if current_permutation[i] > current_permutation[i + 1]:
            current_permutation[i], current_permutation[i + 1] = current_permutation[i + 1], current_permutation[i]
    _, inv_count = count_inversions(current_permutation)
    inversion_counts.append(inv_count)

# Print the results
for count in inversion_counts:
    print(count)