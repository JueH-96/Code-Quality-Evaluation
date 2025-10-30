# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))

# Count inversions for k=0 using merge sort
def merge_and_count(arr, temp, left, mid, right):
    i = left
    j = mid + 1
    k = left
    inv_count = 0
    
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            inv_count += (mid - i + 1)
            j += 1
        k += 1
    
    while i <= mid:
        temp[k] = arr[i]
        i += 1
        k += 1
    
    while j <= right:
        temp[k] = arr[j]
        j += 1
        k += 1
    
    for i in range(left, right + 1):
        arr[i] = temp[i]
    
    return inv_count

def merge_sort_and_count(arr, temp, left, right):
    inv_count = 0
    if left < right:
        mid = (left + right) // 2
        inv_count += merge_sort_and_count(arr, temp, left, mid)
        inv_count += merge_sort_and_count(arr, temp, mid + 1, right)
        inv_count += merge_and_count(arr, temp, left, mid, right)
    return inv_count

# Calculate inversions for each k
results = []
for k in range(m):
    b = [(a[i] + k) % m for i in range(n)]
    temp = [0] * n
    b_copy = b[:]
    inversions = merge_sort_and_count(b_copy, temp, 0, n - 1)
    results.append(inversions)

for r in results:
    print(r)