import sys

def can_transform(A, B, K):
    N = len(A)
    min_queue = []
    max_queue = []
    
    for i in range(N):
        while min_queue and A[min_queue[-1]] >= A[i]:
            min_queue.pop()
        while max_queue and A[max_queue[-1]] <= A[i]:
            max_queue.pop()
        min_queue.append(i)
        max_queue.append(i)
        
        if i >= K:
            if min_queue[0] == i - K:
                min_queue.pop(0)
            if max_queue[0] == i - K:
                max_queue.pop(0)
        
        if i >= K - 1:
            if B[i] < A[min_queue[0]] or B[i] > A[max_queue[0]]:
                return False
    return True

def solve_case(N, K, A, B):
    if can_transform(A, B, K) and can_transform(B, A, K):
        return "Yes"
    return "No"

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N, K = int(data[index]), int(data[index + 1])
        index += 2
        A = list(map(int, data[index:index + N]))
        B = list(map(int, data[index + N:index + 2 * N]))
        index += 2 * N
        results.append(solve_case(N, K, A, B))
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()