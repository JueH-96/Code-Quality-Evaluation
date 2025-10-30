# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    ans = 0
    for k in range(1, n + 1):
        len_pyramid = 2 * k - 1
        if len_pyramid > n:
            break

        possible = False
        for i in range(n - len_pyramid + 1):
            sub_array = a[i:i + len_pyramid]
            
            valid = True
            for j in range(k):
                if sub_array[j] < j + 1:
                    valid = False
                    break
                if sub_array[len_pyramid - 1 - j] < j + 1:
                    valid = False
                    break
            
            if valid:
                possible = True
                break
        
        if possible:
            ans = max(ans,k)

    print(ans)

solve()