# YOUR CODE HERE
import sys

def can_divide(A, N, K, target):
    count = 0
    current_sum = 0
    for i in range(N):
        current_sum += A[i]
        if current_sum >= target:
            count += 1
            current_sum = 0
    return count >= K

def find_min_max(A, N, K):
    total_sum = sum(A)
    low, high = 1, total_sum
    result = 0
    while low <= high:
        mid = (low + high) // 2
        if can_divide(A, N, K, mid):
            result = mid
            low = mid + 1
        else:
            high = mid - 1
    return result

def count_uncut_lines(A, N, K, target):
    count = 0
    current_sum = 0
    uncut_lines = set()
    for i in range(N):
        current_sum += A[i]
        if current_sum >= target:
            count += 1
            current_sum = 0
            if count < K:
                uncut_lines.add(i)
    return len(uncut_lines)

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    target = find_min_max(A, N, K)
    uncut_lines = count_uncut_lines(A, N, K, target)
    
    print(target, uncut_lines)

if __name__ == "__main__":
    main()