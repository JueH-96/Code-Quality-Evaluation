import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+N]))
    idx += N
    K = int(data[idx])
    idx += 1
    queries = []
    for _ in range(K):
        X = int(data[idx])
        Y = int(data[idx+1])
        queries.append((X, Y))
        idx += 2
    
    # Precompute prefix sums for A and B
    A_sorted = sorted(A)
    B_sorted = sorted(B)
    
    # Precompute prefix sums for A_sorted and B_sorted
    prefix_A = [0] * (N + 1)
    for i in range(N):
        prefix_A[i+1] = prefix_A[i] + A_sorted[i]
    
    prefix_B = [0] * (N + 1)
    for i in range(N):
        prefix_B[i+1] = prefix_B[i] + B_sorted[i]
    
    for x, y in queries:
        # Get the first x elements of A_sorted
        A_sub = A_sorted[:x]
        # Get the first y elements of B_sorted
        B_sub = B_sorted[:y]
        
        # Compute the sum of |a - b| for all a in A_sub and b in B_sub
        # Using the formula: sum_{a in A_sub} sum_{b in B_sub} |a - b| = sum_{a in A_sub} (sum_{b <= a} (a - b) + sum_{b > a} (b - a))
        # Which can be rewritten as: sum_{a in A_sub} (a * count_{b <= a} - sum_{b <= a} b + sum_{b > a} b - a * count_{b > a})
        # Or: sum_{a in A_sub} (a * (count_{b <= a} - count_{b > a}) + (sum_{b > a} b - sum_{b <= a} b))
        
        # Precompute the prefix sums for B_sub
        prefix_B_sub = [0] * (y + 1)
        for i in range(y):
            prefix_B_sub[i+1] = prefix_B_sub[i] + B_sub[i]
        
        total = 0
        for a in A_sub:
            # Find the number of elements in B_sub <= a
            cnt_le = bisect.bisect_right(B_sub, a)
            sum_le = prefix_B_sub[cnt_le]
            cnt_gt = y - cnt_le
            sum_gt = prefix_B_sub[y] - sum_le
            # Calculate the sum |a - b| for all b in B_sub
            # sum_{b <= a} (a - b) = a * cnt_le - sum_le
            # sum_{b > a} (b - a) = sum_gt - a * cnt_gt
            total += (a * cnt_le - sum_le) + (sum_gt - a * cnt_gt)
        
        print(total)

if __name__ == "__main__":
    main()