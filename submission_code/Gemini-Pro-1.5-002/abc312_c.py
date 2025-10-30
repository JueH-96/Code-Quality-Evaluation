# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def check(x):
        sellers = 0
        for price in a:
            if price >= x:
                sellers += 1
        
        buyers = 0
        for price in b:
            if price >= x:
                buyers +=1
        return sellers >= buyers
    
    
    left = 1
    right = 10**9 + 1
    ans = -1
    
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
            
    
    
    left_ans = -1
    left = 1
    right = 10**9 + 1
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            left_ans = mid
            right = mid -1
        else:
            left = mid + 1

    print(left_ans)
    

solve()