def find_kth_321_like_number(k):
    """
    Find the k-th smallest 321-like number.
    
    A 321-like number has digits that are strictly decreasing from left to right.
    """
    # Memoization cache for our recursive function
    memo = {}
    
    def count_321_like_numbers(length, first_digit):
        """
        Count the number of 321-like numbers with the specified length
        and first digit exactly first_digit.
        """
        if (length, first_digit) in memo:
            return memo[(length, first_digit)]
        
        if length == 1:
            return 1  # For 1-digit numbers, there's exactly one for each first_digit from 1 to 9
        
        count = 0
        for second_digit in range(first_digit):
            count += count_321_like_numbers(length - 1, second_digit)
        
        memo[(length, first_digit)] = count
        return count
    
    # Determine the length of the k-th 321-like number
    length = 1
    cumulative_count = 0
    
    while True:
        count_for_length = sum(count_321_like_numbers(length, d) for d in range(1, 10))
        if cumulative_count + count_for_length >= k:
            break
        cumulative_count += count_for_length
        length += 1
    
    # Adjust k to be the position within its length category
    k -= cumulative_count
    
    # Find the actual k-th number in its category
    result = []
    
    # Determine the first digit
    for first_digit in range(1, 10):
        count = count_321_like_numbers(length, first_digit)
        if k <= count:
            result.append(first_digit)
            break
        k -= count
    
    # Determine the rest of the digits
    current_length = length - 1
    prev_digit = result[0]
    
    while current_length > 0:
        for digit in range(prev_digit):
            count = count_321_like_numbers(current_length, digit)
            if k <= count:
                result.append(digit)
                prev_digit = digit
                break
            k -= count
        current_length -= 1
    
    return int(''.join(map(str, result)))

if __name__ == "__main__":
    k = int(input())
    print(find_kth_321_like_number(k))