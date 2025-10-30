# YOUR CODE HERE
def nand(a, b):
    return int(not (a and b))

def solve(N, S):
    A = [int(c) for c in S]
    prefix_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]
    
    result = 0
    for i in range(N):
        count_ones = prefix_sum[N] - prefix_sum[i]
        count_zeros = (N - i) - count_ones
        result += count_zeros + (count_ones if A[i] == 0 else 0)
    
    return result

if __name__ == "__main__":
    N = int(input().strip())
    S = input().strip()
    print(solve(N, S))