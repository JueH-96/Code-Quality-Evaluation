def solve():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    X = list(map(int, data[1:N+1]))          # Coordinates
    P = list(map(int, data[N+1:2*N+1]))      # Populations

    # Build a prefix sum of populations
    prefix_sum = [0]*(N+1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + P[i]

    Q = int(data[2*N+1])                    # Number of queries
    answers = []
    pos = 2*N+2

    for _ in range(Q):
        L = int(data[pos]); R = int(data[pos+1])
        pos += 2

        # Find the leftmost index such that X[index]>=L
        left_index = bisect.bisect_left(X, L)
        # Find the rightmost index such that X[index]<=R
        right_index = bisect.bisect_right(X, R)

        if left_index >= right_index:
            answers.append("0")
        else:
            # Sum of populations from left_index to right_index-1
            total_pop = prefix_sum[right_index] - prefix_sum[left_index]
            answers.append(str(total_pop))

    print("
".join(answers))

def main():
    solve()

if __name__ == "__main__":
    main()