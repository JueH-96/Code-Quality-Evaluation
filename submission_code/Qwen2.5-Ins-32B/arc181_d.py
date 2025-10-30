import sys

def read_ints(): return map(int, input().split())
def read_int(): return int(input())
def read_str(): return input().strip()

def count_inversions(arr):
    def merge_sort(arr, temp_arr, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += merge_sort(arr, temp_arr, left, mid)
            inv_count += merge_sort(arr, temp_arr, mid + 1, right)
            inv_count += merge(arr, temp_arr, left, mid, right)
        return inv_count

    def merge(arr, temp_arr, left, mid, right):
        i = left
        j = mid + 1
        k = left
        inv_count = 0

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                i += 1
            else:
                temp_arr[k] = arr[j]
                inv_count += (mid - i + 1)
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

        return inv_count

    n = len(arr)
    temp_arr = [0] * n
    return merge_sort(arr, temp_arr, 0, n - 1)

def bubble_sort_k(arr, k):
    for i in range(1, k):
        if arr[i - 1] > arr[i]:
            arr[i - 1], arr[i] = arr[i], arr[i - 1]

def solve():
    N = read_int()
    P = list(read_ints())
    M = read_int()
    A = list(read_ints())

    inversions = [0] * M
    current_inversions = count_inversions(P)

    for i in range(M):
        k = A[i]
        for j in range(1, k):
            if P[j - 1] > P[j]:
                current_inversions -= 1
                P[j - 1], P[j] = P[j], P[j - 1]
        inversions[i] = current_inversions

    for inv in inversions:
        print(inv)

if __name__ == "__main__":
    solve()