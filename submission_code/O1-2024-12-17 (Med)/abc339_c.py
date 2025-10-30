def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    partial_sum = 0
    min_partial_sum = 0
    
    for val in A:
        partial_sum += val
        if partial_sum < min_partial_sum:
            min_partial_sum = partial_sum
    
    # x is the minimum initial number of passengers so that
    # partial_sum (updated at each stop) never goes negative
    x = max(0, -min_partial_sum)
    
    # Final number of passengers is x plus the sum of all changes
    answer = x + partial_sum
    
    print(answer)

# Do not remove the following line
if __name__ == "__main__":
    main()