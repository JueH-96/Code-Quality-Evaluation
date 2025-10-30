import bisect
import math

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    A.sort()
    answer = 0

    for j in range(1, n):
        x = A[j]
        sqrt_x = int(math.isqrt(x))
        # Compute sum_part1: sum of x//a for a in A[0..j-1] where a <= sqrt_x
        m = bisect.bisect_right(A, sqrt_x, 0, j)
        sum_part1 = 0
        # To optimize, iterate up to m (could be large, but let the code run as is)
        # In practice, this part might be the bottleneck for certain cases
        for a in A[:m]:
            sum_part1 += x // a
        
        # Compute sum_part2: sum of k * count for a > sqrt_x
        sum_part2 = 0
        max_k = sqrt_x
        for k in range(1, max_k + 1):
            L = x / (k + 1)
            R = x / k
            lower_int = int(L) + 1
            upper_int = int(R)
            lower_a = max(lower_int, sqrt_x + 1)
            if lower_a > upper_int:
                continue
            # Find count of elements in [lower_a, upper_int] in A[0..j-1]
            left = bisect.bisect_left(A, lower_a, 0, j)
            right = bisect.bisect_right(A, upper_int, 0, j)
            count = right - left
            sum_part2 += k * count
        
        answer += sum_part1 + sum_part2
    
    print(answer)

if __name__ == "__main__":
    main()