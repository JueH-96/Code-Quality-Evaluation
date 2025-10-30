def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    K = int(data[idx+1])
    idx +=2
    functions = []
    for _ in range(N):
        A = int(data[idx])
        B = int(data[idx+1])
        functions.append((A, B))
        idx +=2
    
    selected = set()
    current_list = []
    prefix_sums = [0]
    suffix_products = [1]
    product_old = 1
    sum_S_old = 0
    
    for _ in range(K):
        max_total = -1
        best_i = -1
        best_p = 0
        for i in range(N):
            if i not in selected:
                A, B = functions[i]
                # Binary search to find insertion position
                low = 0
                high = len(current_list)
                while low < high:
                    mid = (low + high) // 2
                    a_mid, b_mid = current_list[mid]
                    val1 = b_mid * (A - 1)
                    val2 = B * (a_mid - 1)
                    if val1 > val2:
                        low = mid + 1
                    else:
                        high = mid
                p = low
                # Calculate sum_S_new_candidate
                m_old = len(current_list)
                if m_old == 0:
                    sum_S_old_old = 0
                    sum_S_new_candidate = B
                else:
                    sum_S_old_old = prefix_sums[-1]
                    if p >= len(prefix_sums) or p < 0:
                        # This case should not occur due to binary search constraints
                        sum_S_new_candidate = -1
                    else:
                        sum_S_new_candidate = (A - 1) * prefix_sums[p] + sum_S_old_old + B * suffix_products[p]
                total_candidate = product_old * A + sum_S_new_candidate
                if total_candidate > max_total:
                    max_total = total_candidate
                    best_i = i
                    best_p = p
        # Insert the best function
        selected.add(best_i)
        A_best, B_best = functions[best_i]
        current_list.insert(best_p, functions[best_i])
        # Rebuild prefix_sums and suffix_products
        m_new = len(current_list)
        new_suffix = [1] * (m_new + 1)
        for i in range(m_new-1, -1, -1):
            a, b = current_list[i]
            new_suffix[i] = a * new_suffix[i+1]
        new_prefix = [0] * (m_new + 1)
        for i in range(m_new):
            a, b = current_list[i]
            new_prefix[i+1] = new_prefix[i] + b * new_suffix[i+1]
        prefix_sums = new_prefix
        suffix_products = new_suffix
        product_old = suffix_products[0]
        sum_S_old = prefix_sums[-1]
    
    print(product_old + sum_S_old)

if __name__ == "__main__":
    main()