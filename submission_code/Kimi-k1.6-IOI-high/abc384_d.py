def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    S = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    total_sum = sum(A)
    current_sum = 0
    prefix_sums = [0]
    
    # Generate prefix sums for the two-period array
    for num in A:
        current_sum += num
        prefix_sums.append(current_sum)
    for num in A:
        current_sum += num
        prefix_sums.append(current_sum)
    
    residue_map = {0: 0}
    found = False
    for j in range(1, len(prefix_sums)):
        current_P = prefix_sums[j]
        target_residue = (current_P - S) % total_sum
        if target_residue in residue_map:
            max_pi = residue_map[target_residue]
            if current_P - max_pi <= S:
                found = True
                break
        current_residue = current_P % total_sum
        if current_residue not in residue_map or current_P > residue_map.get(current_residue, -1):
            residue_map[current_residue] = current_P
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()