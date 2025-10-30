def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    n = int(input_data[0])
    pairs = []
    index = 1
    for i in range(n):
        L = int(input_data[index])
        R = int(input_data[index+1])
        pairs.append((L, R))
        index += 2

    # Calculate the sum of the minimum bounds and maximum bounds.
    total_min = sum(L for L, R in pairs)
    total_max = sum(R for L, R in pairs)
    
    # For a solution to exist, 0 must be between the sum of mins and maxes.
    if total_min > 0 or total_max < 0:
        sys.stdout.write("No")
        return

    # We'll start with X_i = L_i, and then add a certain amount to reach sum 0.
    delta = -total_min  # additional total we must add to reach 0.
    result = []
    
    for L, R in pairs:
        available = R - L
        add = min(available, delta)
        result.append(L + add)
        delta -= add

    # If delta isn't zero, no solution exists.
    if delta != 0:
        sys.stdout.write("No")
    else:
        output = "Yes
" + " ".join(str(x) for x in result)
        sys.stdout.write(output)

if __name__ == '__main__':
    main()