# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    r = list(map(int, input().split()))
    r.sort()
    
    for _ in range(q):
        x = int(input())
        count = 0
        current_sum = 0
        for i in range(n):
            if current_sum + r[i] <= x:
                current_sum += r[i]
                count += 1
            else:
                break
        print(count)

solve()