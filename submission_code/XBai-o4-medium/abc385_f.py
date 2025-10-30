import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = [0] * N
    H = [0] * N
    for i in range(N):
        X[i] = int(input[ptr])
        H[i] = int(input[ptr+1])
        ptr += 2

    # Compute Previous Greater Element (PGE)
    stack = []
    pge = [None] * N
    for i in range(N):
        while stack and H[stack[-1]] <= H[i]:
            stack.pop()
        if stack:
            pge[i] = stack[-1]
        else:
            pge[i] = None
        stack.append(i)

    # Compute First Smaller Element (FSE)
    stack = []
    fse = [None] * N
    for i in range(N):
        while stack and H[stack[-1]] >= H[i]:
            stack.pop()
        if stack:
            fse[i] = stack[-1]
        else:
            fse[i] = None
        stack.append(i)

    overall_max = -float('inf')

    for i in range(1, N):
        candidates = []
        # Process PGE
        if pge[i] is not None:
            j_high = pge[i]
            x_i_val = X[i]
            H_i_val = H[i]
            x_j = X[j_high]
            H_j = H[j_high]
            numerator = H_j * x_i_val - H_i_val * x_j
            denominator = x_i_val - x_j
            val_high = numerator / denominator
            candidates.append(val_high)
        # Process FSE
        if fse[i] is not None:
            j_low = fse[i]
            x_i_val = X[i]
            H_i_val = H[i]
            x_j = X[j_low]
            H_j = H[j_low]
            numerator = H_j * x_i_val - H_i_val * x_j
            denominator = x_i_val - x_j
            val_low = numerator / denominator
            candidates.append(val_low)
        # Check if all previous are equal
        if pge[i] is None and fse[i] is None and i > 0:
            candidates.append(H[i])
        # Update overall_max
        if len(candidates) > 0:
            current_max = max(candidates)
            if current_max > overall_max:
                overall_max = current_max

    if overall_max < 0:
        print(-1)
    else:
        print("{0:.20f}".format(overall_max))

if __name__ == "__main__":
    main()