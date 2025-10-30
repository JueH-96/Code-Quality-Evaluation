# YOUR CODE HERE
from bisect import bisect_left, bisect_right

def main():
    N = int(input())
    X = list(map(int, input().split()))
    P = list(map(int, input().split()))

    # Create prefix sum array
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + P[i]

    Q = int(input())
    for _ in range(Q):
        L, R = map(int, input().split())
        
        # Find the leftmost and rightmost villages within the range
        left = bisect_left(X, L)
        right = bisect_right(X, R)
        
        # Calculate the sum of villagers in the range
        result = prefix_sum[right] - prefix_sum[left]
        
        print(result)

if __name__ == "__main__":
    main()