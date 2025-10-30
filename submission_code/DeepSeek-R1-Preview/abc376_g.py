import sys
import heapq

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        p_list = list(map(int, data[idx:idx + N]))
        idx += N
        a_list = list(map(int, data[idx:idx + N]))
        idx += N
        
        children = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            parent = p_list[i - 1]
            children[parent].append(i)
        
        s = [0] * (N + 1)
        for i in range(1, N + 1):
            s[i] = a_list[i - 1]
        
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                if node != 0:
                    total = a_list[node - 1]
                    for child in children[node]:
                        total += s[child]
                    s[node] = total
                else:
                    total = 0
                    for child in children[0]:
                        total += s[child]
                    s[0] = total
        
        heap = []
        for child in children[0]:
            heapq.heappush(heap, (-s[child], child))
        
        steps = [0] * (N + 1)
        step = 0
        while heap:
            neg_priority, u = heapq.heappop(heap)
            step += 1
            steps[u] = step
            for child in children[u]:
                heapq.heappush(heap, (-s[child], child))
        
        sum_a = sum(a_list)
        numerator = 0
        for i in range(1, N + 1):
            numerator = (numerator + a_list[i - 1] * steps[i]) % MOD
        
        if sum_a == 0:
            print(0)
        else:
            inv_sum_a = pow(sum_a, MOD - 2, MOD)
            result = (numerator * inv_sum_a) % MOD
            print(result)

if __name__ == '__main__':
    main()