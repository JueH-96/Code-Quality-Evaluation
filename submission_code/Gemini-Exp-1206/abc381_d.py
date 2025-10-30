def is_1122_sequence(arr):
    n = len(arr)
    if n % 2 != 0:
        return False
    counts = {}
    for x in arr:
        counts[x] = counts.get(x, 0) + 1
    for x in arr:
        if counts[x] != 2:
            return False
    for i in range(0, n, 2):
        if arr[i] != arr[i+1]:
            return False
    return True

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_len = 0
    for i in range(n):
        for j in range(i, n):
            sub_array = a[i:j+1]
            if is_1122_sequence(sub_array):
                max_len = max(max_len, len(sub_array))
    
    print(max_len)

solve()