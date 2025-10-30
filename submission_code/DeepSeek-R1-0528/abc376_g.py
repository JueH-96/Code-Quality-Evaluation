import heapq
import sys

mod = 998244353

class HeapNode:
    __slots__ = ('w', 's', 'v')
    def __init__(self, w, s, v):
        self.w = w
        self.s = s
        self.v = v
    def __lt__(self, other):
        prod1 = self.w * other.s
        prod2 = other.w * self.s
        if prod1 != prod2:
            return prod1 > prod2
        return self.w > other.w

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        N = int(data[index]); index += 1
        p_list = list(map(int, data[index:index+N]))
        index += N
        a_list = list(map(int, data[index:index+N]))
        index += N
        
        children = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            parent = p_list[i-1]
            children[parent].append(i)
        
        total_A = sum(a_list)
        
        a_val = [0] * (N+1)
        for i in range(1, N+1):
            a_val[i] = a_list[i-1]
        
        w = [0] * (N+1)
        s_arr = [1] * (N+1)
        
        stack = [0]
        order = []
        while stack:
            u = stack.pop()
            order.append(u)
            for v in children[u]:
                stack.append(v)
                
        for i in range(len(order)-1, -1, -1):
            u = order[i]
            w[u] = a_val[u]
            for v in children[u]:
                w[u] += w[v]
                s_arr[u] += s_arr[v]
                
        heap = []
        for v in children[0]:
            heapq.heappush(heap, HeapNode(w[v], s_arr[v], v))
            
        current_sum = 0
        total_cost = 0
        while heap:
            node = heapq.heappop(heap)
            v = node.v
            total_cost += total_A - current_sum
            current_sum += a_val[v]
            for child in children[v]:
                heapq.heappush(heap, HeapNode(w[child], s_arr[child], child))
                
        total_cost %= mod
        inv_A = pow(total_A, mod-2, mod)
        ans = total_cost * inv_A % mod
        out_lines.append(str(ans))
        
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()