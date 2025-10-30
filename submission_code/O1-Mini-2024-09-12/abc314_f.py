# YOUR CODE HERE
import sys
import sys
from collections import deque

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    p_q = data[1:]
    p = []
    q = []
    for i in range(0, 2*(N-1), 2):
        p.append(int(p_q[i]))
        q.append(int(p_q[i+1]))
        
    total_nodes = 2 * N
    left_child = [0] * (total_nodes +1)
    right_child = [0] * (total_nodes +1)
    size = [1] * (total_nodes +1)
    parent = list(range(total_nodes +1))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    current_node = N +1
    for j in range(N-1):
        p_j = p[j]
        q_j = q[j]
        root_p = find(p_j)
        root_q = find(q_j)
        left_child[current_node] = root_p
        right_child[current_node] = root_q
        size[current_node] = size[root_p] + size[root_q]
        parent[root_p] = current_node
        parent[root_q] = current_node
        current_node +=1
    root = current_node -1
    
    E = [0] * (N +1)
    queue = deque()
    queue.append((root, 0))
    while queue:
        node, run_sum = queue.popleft()
        if node <= N:
            E[node] = run_sum
        else:
            left = left_child[node]
            right = right_child[node]
            a = size[left]
            b = size[right]
            size_ab = a + b
            inv_ab = pow(size_ab, MOD -2, MOD)
            add_left = (run_sum + a * inv_ab) % MOD
            add_right = (run_sum + b * inv_ab) % MOD
            queue.append((left, add_left))
            queue.append((right, add_right))
    result = []
    for i in range(1, N+1):
        result.append(str(E[i]))
    print(' '.join(result))

if __name__ == "__main__":
    main()