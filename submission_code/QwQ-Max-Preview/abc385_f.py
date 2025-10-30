def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    data = list(map(int, input[1:]))
    X = []
    H = []
    for i in range(n):
        X.append(data[2*i])
        H.append(data[2*i+1])
    
    max_ratio = -float('inf')
    j_max = -1
    h_list = []
    for i in range(n):
        X_i = X[i]
        H_i = H[i]
        if i == 0:
            h_i = -float('inf')
        else:
            if j_max == -1:
                h_i = -float('inf')
            else:
                X_j = X[j_max]
                H_j = H[j_max]
                numerator = H_j * X_i - H_i * X_j
                denominator = X_i - X_j
                if denominator == 0:
                    h_i = float('inf')
                else:
                    h_i = numerator / denominator
        h_list.append(h_i)
        
        current_ratio = H_i / X_i
        if current_ratio > max_ratio:
            max_ratio = current_ratio
            j_max = i
    
    M = max(h_list) if h_list else -float('inf')
    
    if M <= 0:
        current_max = -float('inf')
        all_visible = True
        for i in range(n):
            if i == 0:
                continue
            current_ratio = H[i] / X[i]
            if current_ratio <= current_max:
                all_visible = False
                break
            current_max = max(current_max, H[i-1]/X[i-1])
        if all_visible:
            print(-1)
        else:
            print("{0:.20f}".format(M))
    else:
        print("{0:.20f}".format(M))
    
if __name__ == "__main__":
    main()