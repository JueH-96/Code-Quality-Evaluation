# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import sys
    from sys import stdin
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx +=1
    results = []
    for _ in range(T):
        N, K = int(input[idx]), int(input[idx+1])
        idx +=2
        A = list(map(int, input[idx:idx+N]))
        idx +=N
        B = list(map(int, input[idx:idx+N]))
        idx +=N
        items = sorted(zip(A,B), key=lambda x: x[0])
        heap = []
        sum_b =0
        min_value = float('inf')
        for a,b in items:
            heapq.heappush(heap, -b)
            sum_b += b
            if len(heap) > K:
                removed = -heapq.heappop(heap)
                sum_b -= removed
            if len(heap) == K:
                current = a * sum_b
                if current < min_value:
                    min_value = current
        results.append(str(min_value))
    print('
'.join(results))

if __name__ == "__main__":
    main()