import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    a_b = []
    for _ in range(N):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        a_b.append((a, b))
    
    # Sort the list by a_i in ascending order
    sorted_list = sorted(a_b, key=lambda x: x[0])
    a_list = [x[0] for x in sorted_list]
    
    # Create suffix sum array
    sum_b = [0] * (N + 1)
    for i in range(N-1, -1, -1):
        sum_b[i] = sum_b[i+1] + sorted_list[i][1]
    
    # Binary search between left and right
    left = 1
    right = sorted_list[-1][0] + 1  # max a_i + 1
    
    while left < right:
        mid = (left + right) // 2
        pos = bisect.bisect_left(a_list, mid)
        current_sum = sum_b[pos]
        if current_sum <= K:
            right = mid
        else:
            left = mid + 1
    print(left)

if __name__ == "__main__":
    main()