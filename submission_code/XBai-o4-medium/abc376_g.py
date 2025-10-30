import sys
import heapq

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        p_list = list(map(int, input[ptr:ptr+N]))
        ptr += N
        a_list = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Build children list
        children = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            parent = p_list[i-1]
            children[parent].append(i)
        
        # Initialize the priority queue
        heap = []
        for child in children[0]:
            heapq.heappush(heap, (-a_list[child-1], child))
        
        order = [0] * (N + 1)
        step = 0
        
        while heap:
            neg_a, u = heapq.heappop(heap)
            step += 1
            order[u] = step
            for v in children[u]:
                heapq.heappush(heap, (-a_list[v-1], v))
        
        # Calculate numerator and denominator
        S = sum(a_list)
        numerator = 0
        for i in range(1, N + 1):
            numerator += order[i] * a_list[i-1]
        
        inv_S = pow(S, MOD-2, MOD)
        ans = (numerator * inv_S) % MOD
        print(ans)
        
if __name__ == "__main__":
    main()