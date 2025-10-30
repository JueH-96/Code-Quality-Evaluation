# YOUR CODE HERE
def solve():
    n = int(input())
    q = list(map(int, input().split()))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    ans = 0
    for num_a in range(1001):
        temp_q = list(q)
        possible = True
        for i in range(n):
            temp_q[i] -= num_a * a[i]
            if temp_q[i] < 0:
                possible = False
                break
        
        if not possible:
            break

        num_b = 0
        for i in range(n):
            num_b += temp_q[i] // b[i]
        
        
        ans = max(ans, num_a + num_b)
    
    
    for num_b in range(1001):
        temp_q = list(q)
        possible = True
        for i in range(n):
            temp_q[i] -= num_b * b[i]
            if temp_q[i] < 0:
                possible = False
                break
        
        if not possible:
            break

        num_a = 0
        for i in range(n):
            num_a += temp_q[i] // a[i]
        
        ans = max(ans, num_a + num_b)

    print(ans)


solve()