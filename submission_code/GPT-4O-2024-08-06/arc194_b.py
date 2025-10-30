def merge_and_count(arr, temp_arr, left, mid, right):
    i = left    # Starting index for left subarray
    j = mid + 1 # Starting index for right subarray
    k = left    # Starting index to be sorted
    inv_count = 0
    cost = 0

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
        else:
            # There are mid - i inversions, because all elements left to i in the left subarray
            # are greater than arr[j]
            temp_arr[k] = arr[j]
            inv_count += (mid - i + 1)
            cost += (mid - i + 1) * i
            j += 1
        k += 1

    while i <= mid:
        temp_arr[k] = arr[i]
        i += 1
        k += 1

    while j <= right:
        temp_arr[k] = arr[j]
        j += 1
        k += 1

    for i in range(left, right + 1):
        arr[i] = temp_arr[i]

    return inv_count, cost

def merge_sort_and_count(arr, temp_arr, left, right):
    inv_count = 0
    cost = 0
    if left < right:
        mid = (left + right) // 2

        inv_count_left, cost_left = merge_sort_and_count(arr, temp_arr, left, mid)
        inv_count_right, cost_right = merge_sort_and_count(arr, temp_arr, mid + 1, right)
        inv_count_merge, cost_merge = merge_and_count(arr, temp_arr, left, mid, right)

        inv_count = inv_count_left + inv_count_right + inv_count_merge
        cost = cost_left + cost_right + cost_merge

    return inv_count, cost

def minimum_cost_to_sort_permutation(N, P):
    temp_arr = [0] * N
    _, cost = merge_sort_and_count(P, temp_arr, 0, N - 1)
    return cost

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    P = list(map(int, data[1:]))
    result = minimum_cost_to_sort_permutation(N, P)
    print(result)