def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    partial_sum = 0
    min_partial_sum = 0
    
    # Calculate partial sums and track the minimum partial sum
    for value in A:
        partial_sum += value
        if partial_sum < min_partial_sum:
            min_partial_sum = partial_sum
    
    # The initial passengers must offset any negative partial sum
    initial_passengers = max(0, -min_partial_sum)
    
    # Final number of passengers is the total sum plus the initial offset
    final_passengers = partial_sum + initial_passengers
    
    print(final_passengers)

# Do not remove this call
main()