# YOUR CODE HERE
def is_square(n):
    if n < 0:
        return False
    x = int(n**0.5)
    return x * x == n

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if is_square(a[i] * a[j]):
                count += 1
    print(count)

solve()