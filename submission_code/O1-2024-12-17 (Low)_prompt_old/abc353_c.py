def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    T = 10**8

    # 1) Compute the sum of f(A_i, A_j) if we didn't have the modulus condition,
    #    i.e. sum of (A_i + A_j) over all 1 <= i < j <= N.
    #    We will call this sumPairs.
    #    sumPairs = Σ (A_i + A_j) for i < j.
    #    An O(N) way: for j from 1..N-1 (0-based in code), add prefix[j-1] + j*A[j].
    prefix = [0] * N
    prefix[0] = A[0]
    for i in range(1, N):
        prefix[i] = prefix[i-1] + A[i]

    sumPairs = 0
    for j in range(1, N):
        sumPairs += prefix[j-1] + j * A[j]

    # 2) Count how many pairs (i, j) (i < j) have A_i + A_j >= T.
    #    We'll do a classic two-pointer approach to count how many pairs have sum < T,
    #    then subtract from total pairs.
    A.sort()
    left, right = 0, N - 1
    count_less = 0  # number of pairs with sum < T

    while left < right:
        if A[left] + A[right] < T:
            # all pairs (left, any index from left+1..right) have sum < T
            count_less += (right - left)
            left += 1
        else:
            right -= 1

    total_pairs = N * (N - 1) // 2
    count_large = total_pairs - count_less  # pairs with sum >= T

    # 3) The final answer is sumPairs - (count of pairs that exceed or meet T)*T
    #    Because f(x,y) = (x+y) mod T = (x+y) - T if x+y >= T, else x+y if x+y < T.
    answer = sumPairs - count_large * T

    print(answer)

def main():
    solve()

# Call solve() according to the requested format.
if __name__ == "__main__":
    solve()