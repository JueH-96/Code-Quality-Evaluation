# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def is_pyramid(arr, k):
        if len(arr) != 2 * k - 1:
            return False
        for i in range(k):
            if arr[i] < i + 1:
                return False
        for i in range(k - 1):
            if arr[k + i] < k - 1 - i:
                return False
        return True

    def check(k, arr):
        for i in range(n - (2 * k - 1) + 1):
            sub_arr = arr[i:i + 2 * k - 1]
            
            possible = True
            for j in range(k):
                if sub_arr[j] < j + 1:
                    possible = False
                    break
            if not possible:
                continue
            
            for j in range(k - 1):
                if sub_arr[k + j] < k - 1 - j:
                    possible = False
                    break
            if not possible:
                continue
            
            return True
        return False

    ans = 0
    for k in range(1, n + 2):
        if 2 * k - 1 > n:
            break
        if check(k, a):
            ans = max(ans, k)

    print(ans)

solve()