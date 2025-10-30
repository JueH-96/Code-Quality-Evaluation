# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edge_weights = []
    total_numbers = N * (N - 1) // 2
    while len(edge_weights) < total_numbers:
        line = sys.stdin.readline()
        if not line:
            break
        numbers = line.strip().split()
        for x in numbers:
            if x:
                edge_weights.append(int(x))
    D = [[0]*N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(i+1,N):
            D[i][j] = edge_weights[k]
            k +=1
    memo = {}
    def dfs(mask):
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]
        max_value = 0
        # Find the first set bit i
        for i in range(N):
            if mask & (1<<i):
                break
        for j in range(i+1,N):
            if mask & (1<<j):
                next_mask = mask ^ (1<<i) ^ (1<<j)
                value = dfs(next_mask) + D[i][j]
                if value > max_value:
                    max_value = value
        memo[mask] = max_value
        return max_value
    full_mask = (1<<N)-1
    result = dfs(full_mask)
    print(result)
threading.Thread(target=main).start()
# END CODE