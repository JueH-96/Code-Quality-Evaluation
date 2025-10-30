import heapq

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        p = list(map(int, data[idx:idx+N]))
        idx += N
        a = list(map(int, data[idx:idx+N]))
        idx += N
        
        children = [[] for _ in range(N+1)]  # children[0] to children[N]
        for i in range(N):
            parent = p[i]
            child = i + 1  # nodes are 1-based
            children[parent].append(child)
        
        heap = []
        for child in children[0]:
            # a is 0-based, child is 1-based, so a[child-1]
            heapq.heappush(heap, (-a[child-1], child))
        
        sum_pos = 0
        total_a = sum(a)
        pos = 1
        
        while heap:
            neg_a, u = heapq.heappop(heap)
            a_u = -neg_a
            sum_pos = (sum_pos + a_u * pos) % MOD
            for v in children[u]:
                heapq.heappush(heap, (-a[v-1], v))
            pos += 1
        
        inv_total_a = pow(total_a, MOD-2, MOD)
        ans = (sum_pos * inv_total_a) % MOD
        print(ans)

if __name__ == "__main__":
    main()