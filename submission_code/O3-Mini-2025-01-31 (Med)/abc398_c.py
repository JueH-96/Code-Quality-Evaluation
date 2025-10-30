def main():
    import sys
    input = sys.stdin.readline
    
    # Read input values
    n = int(input().strip())
    a = list(map(int, input().split()))
    
    # Count frequency of each number
    freq = {}
    for number in a:
        freq[number] = freq.get(number, 0) + 1
    
    # Find the one with unique occurrence and maximum value
    max_value = -1
    label = -1
    for idx, number in enumerate(a):
        # Check unique occurrence and update if this number is greater than current maximum
        if freq[number] == 1 and number > max_value:
            max_value = number
            label = idx + 1  # because person labels are 1-indexed
    
    # Print the result
    print(label)

if __name__ == '__main__':
    main()