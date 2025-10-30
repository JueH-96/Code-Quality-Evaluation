def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    intervals = input_data[2:]
    lr = [(int(intervals[2*i]), int(intervals[2*i+1])) for i in range(N)]
    
    # Sort intervals by R_i
    lr.sort(key=lambda x: x[1])

    # We will maintain maxL[r] = maximum L_i among all intervals whose R_i <= r.
    # If no interval satisfies R_i <= r, maxL[r] = 0.
    maxL = [0]*(M+1)

    idx = 0  # index in sorted intervals
    current_max_L = 0

    for r in range(1, M+1):
        # Move idx forward while intervals[idx].R <= r
        while idx < N and lr[idx][1] <= r:
            current_max_L = max(current_max_L, lr[idx][0])
            idx += 1
        maxL[r] = current_max_L

    # Sum of min(r, maxL[r]) over r=1..M
    union_count = 0
    for r in range(1, M+1):
        union_count += min(r, maxL[r])

    # Total number of (l,r) with 1<=l<=r<=M is M*(M+1)//2
    total_pairs = M*(M+1)//2
    answer = total_pairs - union_count

    print(answer)

# Do not forget to call main()
main()