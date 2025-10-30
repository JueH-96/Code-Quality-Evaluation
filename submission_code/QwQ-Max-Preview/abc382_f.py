def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx += 1
    W = int(input[idx]); idx += 1
    N = int(input[idx]); idx += 1

    bars = []
    for _ in range(N):
        R = int(input[idx]); idx += 1
        C = int(input[idx]); idx += 1
        L = int(input[idx]); idx += 1
        bars.append((R, C, L))
    
    earliest_row = [0] * (W + 2)  # 1-based indexing for columns

    output = []
    for R, C, L in reversed(bars):
        min_row = H
        for c in range(C, C + L):
            if earliest_row[c] == 0:
                candidate = H
            else:
                candidate = earliest_row[c] - 1
            if candidate < min_row:
                min_row = candidate
        final_row = min(H, min_row)
        # Update the earliest_row for each column in the bar's span
        for c in range(C, C + L):
            if earliest_row[c] < final_row:
                earliest_row[c] = final_row
        output.append(final_row)
    
    # Reverse the output to restore the original order
    output.reverse()
    for r in output:
        print(r)

if __name__ == '__main__':
    main()