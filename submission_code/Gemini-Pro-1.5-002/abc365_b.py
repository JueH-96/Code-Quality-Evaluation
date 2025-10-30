# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    indexed_a = list(enumerate(a, 1))
    indexed_a.sort(key=lambda x: x[1], reverse=True)
    
    print(indexed_a[1][0])

solve()