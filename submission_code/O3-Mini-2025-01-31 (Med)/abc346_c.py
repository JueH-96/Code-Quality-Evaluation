def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    
    # Calculate the total sum of integers from 1 to K using the formula
    total_sum = K * (K + 1) // 2
    
    # Use a set to collect unique elements from A that are between 1 and K (inclusive)
    unique_in_range = set()
    for number in A:
        if 1 <= number <= K:
            unique_in_range.add(number)
    
    # Sum the numbers that are present in the list (and in the range)
    present_sum = sum(unique_in_range)
    
    # The answer is the difference between the total sum and the present_sum
    missing_sum = total_sum - present_sum
    print(missing_sum)
    
if __name__ == "__main__":
    main()