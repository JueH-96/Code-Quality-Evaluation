# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    a = list(map(int, data[3:3+N]))
    b = list(map(int, data[3+N:3+N+M]))
    
    forbidden = set()
    index = 3 + N + M
    for _ in range(L):
        c = int(data[index])
        d = int(data[index+1])
        forbidden.add((c, d))
        index += 2
    
    # Find the maximum a and b
    max_a = max(a)
    max_b = max(b)
    
    # Check if the combination (max_a, max_b) is forbidden
    # To find the indices of max_a and max_b
    # Since a and b are lists, we need to find the indices where a[i] == max_a and b[j] == max_b
    # But since there can be multiple, we need to find all possible pairs
    
    # Get all indices where a[i] == max_a
    max_a_indices = [i+1 for i, val in enumerate(a) if val == max_a]
    # Get all indices where b[j] == max_b
    max_b_indices = [j+1 for j, val in enumerate(b) if val == max_b]
    
    # Check if any pair (c, d) in max_a_indices x max_b_indices is not in forbidden
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
    else:
        # Need to find the next best combination
        # Sort a and b in descending order
        sorted_a = sorted(a, reverse=True)
        sorted_b = sorted(b, reverse=True)
        
        # Iterate through the top elements to find the maximum allowed combination
        max_price = 0
        for i in range(len(sorted_a)):
            for j in range(len(sorted_b)):
                # Find the indices of sorted_a[i] and sorted_b[j] in the original lists
                # Since sorted_a[i] is the i-th largest in a, and similarly for b
                # We need to find all indices where a[k] == sorted_a[i] and b[l] == sorted_b[j]
                # Then check if any pair (k+1, l+1) is not in forbidden
                a_val = sorted_a[i]
                b_val = sorted_b[j]
                a_indices = [k+1 for k, val in enumerate(a) if val == a_val]
                b_indices = [l+1 for l, val in enumerate(b) if val == b_val]
                for c in a_indices:
                    for d in b_indices:
                        if (c, d) not in forbidden:
                            current_price = a_val + b_val
                            if current_price > max_price:
                                max_price = current_price
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        print(max_price)

if __name__ == "__main__":
    main()