import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    sorted_A = sorted([(A[i], i+1) for i in range(N)], key=lambda x: x[0])
    n = N
    for i in range(n-2):
        a_val = sorted_A[i][0]
        a_idx = sorted_A[i][1]
        # Check minimal sum for this i
        min_sum = a_val + sorted_A[i+1][0] + sorted_A[i+2][0]
        if min_sum > X:
            break
        # Check max sum for this i
        max_sum = a_val + sorted_A[-1][0] + sorted_A[-2][0]
        if max_sum < X:
            continue
        left = i + 1
        right = n - 1
        while left < right:
            current_sum = a_val + sorted_A[left][0] + sorted_A[right][0]
            if current_sum == X:
                indices = [a_idx, sorted_A[left][1], sorted_A[right][1]]
                indices.sort()
                print(f"{indices[0]} {indices[1]} {indices[2]}")
                return
            elif current_sum < X:
                left += 1
            else:
                right -= 1
    print(-1)

if __name__ == "__main__":
    main()