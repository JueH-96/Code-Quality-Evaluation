def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    L = int(data[idx+2])
    idx += 3
    
    a = list(map(int, data[idx:idx+N]))
    idx += N
    
    b = list(map(int, data[idx:idx+M]))
    idx += M
    
    forbidden = set()
    for _ in range(L):
        c = int(data[idx])
        d = int(data[idx+1])
        forbidden.add((c, d))
        idx += 2
    
    # Find the maximum a and b
    max_a = max(a)
    max_b = max(b)
    
    # Check if the combination (max_a, max_b) is forbidden
    # Find the indices of max_a and max_b
    # Since a and b are 1-based in the forbidden set, we need to find the 1-based indices
    # a is 0-based in the list, so a[i] corresponds to main dish i+1
    # Similarly for b
    
    # Find all indices where a[i] == max_a
    max_a_indices = [i+1 for i, val in enumerate(a) if val == max_a]
    # Find all indices where b[j] == max_b
    max_b_indices = [j+1 for j, val in enumerate(b) if val == max_b]
    
    # Check if any combination of max_a_indices and max_b_indices is not forbidden
    found = False
    for c in max_a_indices:
        for d in max_b_indices:
            if (c, d) not in forbidden:
                found = True
                break
        if found:
            break
    
    if found:
        print(max_a + max_b)
        return
    
    # If not, we need to find the next best combination
    # We need to find the maximum a_i + b_j where (i+1, j+1) is not forbidden
    # To optimize, we can sort a and b in descending order and try combinations until we find a valid one
    
    # Sort a and b in descending order
    sorted_a = sorted([(val, i+1) for i, val in enumerate(a)], reverse=True, key=lambda x: x[0])
    sorted_b = sorted([(val, j+1) for j, val in enumerate(b)], reverse=True, key=lambda x: x[0])
    
    # Try the top combinations
    max_price = 0
    for val_a, i in sorted_a:
        for val_b, j in sorted_b:
            if (i, j) not in forbidden:
                current_price = val_a + val_b
                if current_price > max_price:
                    max_price = current_price
                break  # Since sorted, no need to check further for this a_i
        if max_price == max_a + max_b:
            break  # Since we already checked the maximum possible
    
    print(max_price)

if __name__ == "__main__":
    main()