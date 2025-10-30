def count_inversions(arr):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count

def apply_operation(arr, k):
    for i in range(k - 1):
        if arr[i] > arr[i+1]:
            arr[i], arr[i+1] = arr[i+1], arr[i]
    return arr

def solve():
    n = int(input())
    p = list(map(int, input().split()))
    m = int(input())
    a = list(map(int, input().split()))

    for i in range(m):
        for j in range(i+1):
            p = apply_operation(p, a[j])
        print(count_inversions(p))
        
        temp_p = list(map(int, input().split()))
        for k in range(i+1):
            temp_p = apply_operation(temp_p, a[k])
        p = temp_p
        
if __name__ == "__main__":
    solve()