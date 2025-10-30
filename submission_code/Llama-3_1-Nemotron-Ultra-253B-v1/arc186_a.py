def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    Q = int(input[idx])
    idx +=1
    Ks = [int(input[idx + i]) for i in range(Q)]
    
    possible = set()
    
    for a in range(N+1):
        for b in range(N+1):
            # Generate all possible sum_row_a and sum_col_b
            sum_row_a_list = [k * N for k in range(a+1)]
            sum_col_b_list = [l * N for l in range(b+1)]
            found = False
            for sum_row_a in sum_row_a_list:
                # Compute sum_row range
                rest_row_min = (N - a) * 1
                rest_row_max = (N - a) * (N-1)
                sum_row_min = sum_row_a + rest_row_min
                sum_row_max = sum_row_a + rest_row_max
                if sum_row_min > sum_row_max:
                    continue
                for sum_col_b in sum_col_b_list:
                    # Compute sum_col range
                    rest_col_min = (N - b) * 1
                    rest_col_max = (N - b) * (N-1)
                    sum_col_min = sum_col_b + rest_col_min
                    sum_col_max = sum_col_b + rest_col_max
                    if sum_col_min > sum_col_max:
                        continue
                    # Check overlap
                    if not (sum_row_max < sum_col_min or sum_col_max < sum_row_min):
                        possible.add(a * N + b * N - a * b)
                        found = True
                        break
                if found:
                    break
    
    for k in Ks:
        print("Yes" if k in possible else "No")

if __name__ == '__main__':
    main()