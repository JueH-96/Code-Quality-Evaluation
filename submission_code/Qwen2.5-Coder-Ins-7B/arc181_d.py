import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:N+1]))
M = int(data[N+1])
A = list(map(int, data[N+2:N+2+M]))

# Function to calculate the inversion number using merge sort
def merge_sort_and_count(arr):
    if len(arr) <= 1:
        return arr, 0
    mid = len(arr) // 2
    left, left_count = merge_sort_and_count(arr[:mid])
    right, right_count = merge_sort_and_count(arr[mid:])
    merged, split_count = merge_and_count(left, right)
    return merged, left_count + right_count + split_count

def merge_and_count(left, right):
    result = []
    i = j = inv_count = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
            inv_count += len(left) - i
    result.extend(left[i:])
    result.extend(right[j:])
    return result, inv_count

# Process each operation A_i
results = []
current_inversion = 0
for a in A:
    # Apply operations up to A_i
    for k in range(2, a+1):
        for i in range(1, k):
            if P[i-1] > P[i]:
                P[i-1], P[i] = P[i], P[i-1]
                current_inversion += 1
    # Calculate the inversion number after applying operations
    _, current_inversion = merge_sort_and_count(P)
    results.append(current_inversion)

# Print the results
for result in results:
    print(result)