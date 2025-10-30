# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import heapq
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    W = list(map(int, input[N+1:2*N+1]))
    
    box_contents = [[] for _ in range(N+1)]
    for i in range(N):
        heapq.heappush(box_contents[A[i]], W[i])
    
    total_cost = 0
    for box in box_contents:
        if len(box) > 1:
            while len(box) > 1:
                heapq.heappop(box)
            total_cost += box[0]
    
    print(total_cost)

if __name__ == "__main__":
    main()