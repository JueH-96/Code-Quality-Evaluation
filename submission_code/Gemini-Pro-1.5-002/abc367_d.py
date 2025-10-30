# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_sum = sum(a)
    
    count = 0
    for s in range(n):
        current_sum = 0
        for t in range(s + 1, s + n):
            current_sum += a[(t - 1) % n]
            if current_sum % m == 0:
                count += 1
    
    print(count)

solve()