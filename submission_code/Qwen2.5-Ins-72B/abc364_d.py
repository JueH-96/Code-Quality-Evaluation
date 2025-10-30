# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    A = list(map(int, data[2:2+N]))
    queries = [list(map(int, data[2+N+2*i:2+N+2*i+2])) for i in range(Q)]
    
    for b, k in queries:
        distances = [abs(a - b) for a in A]
        distances.sort()
        print(distances[k-1])

if __name__ == "__main__":
    solve()