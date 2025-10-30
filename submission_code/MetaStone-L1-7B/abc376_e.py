import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N, K = int(data[idx]), int(data[idx+1])
        idx +=2
        A = list(map(int, data[idx:idx+N]))
        idx +=N
        B = list(map(int, data[idx:idx+N]))
        idx +=N
        
        sorted_list = sorted(zip(A, B), key=lambda x: x[0])
        A_sorted = [x[0] for x in sorted_list]
        B_sorted = [x[1] for x in sorted_list]
        
        heap = []
        sum_k_minus_1 = 0
        min_product = float('inf')
        
        for i in range(len(sorted_list)):
            if i > 0:
                b = B_sorted[i-1]
                heapq.heappush(heap, b)
                if len(heap) > K - 1:
                    popped = heapq.heappop(heap)
                    sum_k_minus_1 -= popped
            if i >= K - 1:
                current_b = B_sorted[i]
                current_sum = sum_k_minus_1 + current_b
                current_product = A_sorted[i] * current_sum
                if current_product < min_product:
                    min_product = current_product
        
        print(min_product)

if __name__ == '__main__':
    main()