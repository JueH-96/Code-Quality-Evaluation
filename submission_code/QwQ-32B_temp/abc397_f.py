import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    if N < 3:
        print(0)
        return
    
    # Compute left array
    left = [0] * N
    current_set = set()
    current_count = 0
    current_set.add(A[0])
    current_count = 1
    left[0] = current_count
    for i in range(1, N):
        if A[i] not in current_set:
            current_count += 1
            current_set.add(A[i])
        left[i] = current_count
    
    # Compute right array
    right = [0] * N
    current_set = set()
    current_count = 0
    current_set.add(A[-1])
    current_count = 1
    right[-1] = current_count
    for i in range(N-2, -1, -1):
        if A[i] not in current_set:
            current_count += 1
            current_set.add(A[i])
        right[i] = current_count
    
    current_elements = defaultdict(int)
    current_distinct = 0
    max_total = 0
    left_start = 0
    
    for j in range(1, N-1):
        a = A[j]
        if current_elements[a] == 0:
            current_distinct += 1
        current_elements[a] += 1
        
        current_candidate = left[left_start] + current_distinct + right[j+1]
        if current_candidate > max_total:
            max_total = current_candidate
        
        while True:
            next_i = left_start + 1
            if next_i >= j:
                break
            element = A[left_start + 1]
            current_elements[element] -= 1
            if current_elements[element] == 0:
                current_distinct -= 1
            new_candidate = left[next_i] + current_distinct + right[j+1]
            if new_candidate > current_candidate:
                current_candidate = new_candidate
                left_start = next_i
            else:
                current_elements[element] += 1
                if current_elements[element] == 1:
                    current_distinct += 1
                break
        
        if current_candidate > max_total:
            max_total = current_candidate
    
    print(max_total)

if __name__ == "__main__":
    main()