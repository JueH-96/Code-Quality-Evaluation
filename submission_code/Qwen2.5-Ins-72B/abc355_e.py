# YOUR CODE HERE
import sys

def query(i, j):
    print(f"? {i} {j}")
    sys.stdout.flush()
    return int(input())

def solve(N, L, R):
    def get_sum(l, r):
        if l > r:
            return 0
        if (l, r) in memo:
            return memo[(l, r)]
        
        i = 0
        while (1 << (i + 1)) <= (r - l + 1):
            i += 1
        
        j1 = l // (1 << i)
        j2 = r // (1 << i)
        
        if j1 == j2:
            result = query(i, j1)
        else:
            result = (get_sum(l, (j1 + 1) * (1 << i) - 1) + get_sum(j2 * (1 << i), r)) % 100
        
        memo[(l, r)] = result
        return result
    
    memo = {}
    result = get_sum(L, R)
    print(f"! {result}")
    sys.stdout.flush()

N, L, R = map(int, input().split())
solve(N, L, R)