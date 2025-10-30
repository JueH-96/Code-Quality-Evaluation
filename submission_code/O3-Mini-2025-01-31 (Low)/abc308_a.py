def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Convert each to integer.
    try:
        numbers = list(map(int, input_data))
    except:
        print("No")
        return
        
    # If we don't have exactly 8 numbers, it's an error.
    if len(numbers) != 8:
        print("No")
        return
    
    # Unpack the numbers for clarity.
    s1, s2, s3, s4, s5, s6, s7, s8 = numbers
    
    # Condition 1: Monotone non-decreasing.
    monotonically_non_decreasing = (s1 <= s2 <= s3 <= s4 <= s5 <= s6 <= s7 <= s8)
    
    # Condition 2: Every integer must be in the range 100 to 675.
    in_range = all(100 <= s <= 675 for s in numbers)
    
    # Condition 3: Every integer must be a multiple of 25.
    multiples_of_25 = all(s % 25 == 0 for s in numbers)
    
    if monotonically_non_decreasing and in_range and multiples_of_25:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()