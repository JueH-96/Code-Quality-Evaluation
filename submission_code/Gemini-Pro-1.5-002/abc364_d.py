# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    
    for _ in range(q):
        b, k = map(int, input().split())
        distances = []
        for x in a:
            distances.append(abs(x - b))
        distances.sort()
        print(distances[k-1])

solve()