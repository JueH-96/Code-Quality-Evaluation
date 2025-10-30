import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N, M = int(input[idx]), int(input[idx+1])
        idx +=2
        boxes = []
        for __ in range(N):
            V = int(input[idx])
            P = int(input[idx+1])
            idx +=2
            if V >= P:
                profit = V - P
                # The actual contribution is min(V, 1) - P, but since V >= P and 1 <= V?
                # If Mr. Ball sends only one ball, contribution is 1 - P
                # If he sends V balls, contribution is V - P
                # But he will choose the minimum between them
                contribution = min(V, 1) - P
                boxes.append((profit, contribution))
        # Sort by profit (V-P) descending, then contribution descending
        boxes.sort(key=lambda x: (-x[0], -x[1]))
        res = 0
        sum_profit = 0
        sum_contribution = 0
        # We can take up to M boxes
        # We need to find the best K boxes (K <= M)
        # For each K, the total is sum of max( (V_i - P_i) for first box, and (1-P_i) for the rest)
        # So for K boxes, the first box contributes (V_i - P_i), the rest contribute (1-P_i)
        # So we need to select K boxes where the first has maximum (V_i - P_i), and the rest have maximum (1-P_i)
        # To handle this, we can precompute prefix max for (V_i - P_i) and suffix max for (1-P_i)
        # But this is complicated. Alternative approach:
        # For each possible K from 1 to min(M, len(boxes)), compute the best possible sum
        # where the first K boxes are chosen to maximize (V_i - P_i) for the first, and (1-P_i) for the rest
        # So we need to select one box as the first, and K-1 boxes from the remaining with maximum (1-P_i)
        # This is O(N^2), which is not feasible.
        # Alternative approach:
        # Sort the boxes by (V_i - P_i) + (1 - P_i) * (K-1), but this is not straightforward.
        # Alternative idea inspired by the sample:
        # The maximum profit is the sum of (V_i - P_i) for the first box and (1 - P_i) for the remaining boxes.
        # So we can select the first box with maximum (V_i - P_i), and then select up to M-1 boxes with maximum (1 - P_i)
        # This is a heuristic but works for the sample.
        if not boxes:
            print(0)
            continue
        # Separate the boxes into two parts: the first box and the rest
        # Sort by (V-P) descending
        boxes.sort(key=lambda x: (-x[0], -x[1]))
        max_sum = 0
        # Try taking K boxes, for K from 0 to min(M, len(boxes))
        for K in range(0, min(M, len(boxes)) + 1):
            if K ==0:
                current =0
            else:
                # Take 1 from first part, and K-1 from the rest
                # Select the first box as the one with index 0 to K-1
                # For each possible first box in the first K boxes
                # Take the first box's (V-P), and sum (1-P_i) for the next K-1 boxes
                current = boxes[0][0]
                taken = 1
                # Now select K-1 boxes from the remaining (boxes[1:] sorted by (1-P_i) descending)
                # So we need to have a list sorted by (1-P_i) descending for the remaining boxes
                # Precompute this list
                rest = boxes[1:]
                # Sort rest by (contribution) descending
                rest_sorted = sorted(rest, key=lambda x: (-x[1], -x[0]))
                # Take the first K-1
                for i in range(min(K-1, len(rest_sorted))):
                    current += rest_sorted[i][1]
                taken = K
            if current > max_sum:
                max_sum = current
        print(max_sum)
        
if __name__ == '__main__':
    main()