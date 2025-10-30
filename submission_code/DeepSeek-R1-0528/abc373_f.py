import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data: 
        print(0)
        return
    n = int(data[0])
    W = int(data[1])
    w = []
    v = []
    idx = 2
    for i in range(n):
        w_i = int(data[idx])
        v_i = int(data[idx+1])
        idx += 2
        w.append(w_i)
        v.append(v_i)
    
    heap = []
    k = [0] * n
    
    for i in range(n):
        gain0 = v[i] - 1
        if gain0 > 0:
            heapq.heappush(heap, (-gain0, i))
    
    total_gain = 0
    current_weight = 0
    
    while heap and current_weight < W:
        neg_g, i = heapq.heappop(heap)
        gain = -neg_g
        if current_weight + w[i] <= W:
            total_gain += gain
            current_weight += w[i]
            k[i] += 1
            next_g = v[i] - (2 * k[i] + 1)
            if next_g > 0:
                heapq.heappush(heap, (-next_g, i))
                
    print(total_gain)

if __name__ == '__main__':
    main()