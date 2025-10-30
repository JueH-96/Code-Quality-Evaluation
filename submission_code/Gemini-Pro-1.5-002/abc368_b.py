# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    while True:
        positive_count = 0
        for x in a:
            if x > 0:
                positive_count += 1
        
        if positive_count <= 1:
            break
        
        a.sort(reverse=True)
        a[0] -= 1
        a[1] -= 1
        count += 1
        
    print(count)

solve()