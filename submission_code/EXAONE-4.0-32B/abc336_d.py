import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 0:
        print(0)
        return
        
    X = [A[i] - i for i in range(n)]
    Y = [A[i] + i for i in range(n)]
    
    log_table = [0] * (n + 1)
    if n >= 2:
        for i in range(2, n + 1):
            log_table[i] = log_table[i // 2] + 1

    LOG = n.bit_length()
    st_X = [X[:]]
    st_Y = [Y[:]]
    
    for j in range(1, LOG):
        step = 1 << (j - 1)
        prev_level_X = st_X[j - 1]
        prev_level_Y = st_Y[j - 1]
        new_len = n - (1 << j) + 1
        if new_len <= 0:
            break
        curr_X = []
        curr_Y = []
        for i in range(new_len):
            curr_X.append(min(prev_level_X[i], prev_level_X[i + step]))
            curr_Y.append(min(prev_level_Y[i], prev_level_Y[i + step]))
        st_X.append(curr_X)
        st_Y.append(curr_Y)
    
    max_possible_k = 0
    for center in range(n):
        low = 1
        high = min(center + 1, n - center)
        best_k = 0
        while low <= high:
            k = (low + high) // 2
            left_start = center - k + 1
            left_end = center
            j = log_table[k]
            seg_len = k
            if seg_len == 1:
                min_left_val = X[center]
            else:
                idx1 = left_start
                idx2 = left_end - (1 << j) + 1
                min_left_val = min(st_X[j][idx1], st_X[j][idx2])
            condition_left = (min_left_val + center >= k)
            
            right_start = center
            right_end = center + k - 1
            if seg_len == 1:
                min_right_val = Y[center]
            else:
                j = log_table[k]
                idx1 = right_start
                idx2 = right_end - (1 << j) + 1
                min_right_val = min(st_Y[j][idx1], st_Y[j][idx2])
            condition_right = (min_right_val - center >= k)
            
            if condition_left and condition_right:
                best_k = k
                low = k + 1
            else:
                high = k - 1
                
        if best_k > max_possible_k:
            max_possible_k = best_k
            
    print(max_possible_k)

if __name__ == "__main__":
    main()