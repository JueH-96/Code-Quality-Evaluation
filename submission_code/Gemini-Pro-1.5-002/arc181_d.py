# YOUR CODE HERE
def count_inversions(arr):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count

def solve():
    n = int(input())
    p = list(map(int, input().split()))
    m = int(input())
    a = list(map(int, input().split()))

    for k_val in a:
        for k in range(2, k_val + 1):
            for i in range(k - 1):
                if p[i] > p[i + 1]:
                    p[i], p[i + 1] = p[i + 1], p[i]
        print(count_inversions(p))

solve()