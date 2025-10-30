# YOUR CODE HERE
import sys

def find_min_x(n, m, a, b):
    low, high = 1, 10**9
    result = high
    
    while low <= high:
        mid = (low + high) // 2
        sellers = sum(1 for price in a if price <= mid)
        buyers = sum(1 for price in b if price >= mid)
        
        if sellers >= buyers:
            result = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return result

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:]))

print(find_min_x(N, M, A, B))