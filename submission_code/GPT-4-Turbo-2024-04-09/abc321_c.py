def generate_321_like_numbers():
    from heapq import heappush, heappop
    
    # Min-heap to store the numbers in increasing order
    heap = []
    
    # Start by pushing all single digit numbers (1 to 9)
    for i in range(1, 10):
        heappush(heap, i)
    
    # This will store the generated 321-like numbers
    result = []
    
    # Generate numbers until we have enough in the result
    while heap:
        # Get the smallest number from the heap
        smallest = heappop(heap)
        result.append(smallest)
        
        # Get the last digit of the smallest number
        last_digit = smallest % 10
        
        # We can only append smaller digits to ensure the number is strictly decreasing
        for new_digit in range(last_digit):
            new_number = smallest * 10 + new_digit
            heappush(heap, new_number)
    
    return result

def main():
    import sys
    input = sys.stdin.read
    K = int(input().strip())
    
    # Pre-generate a large list of 321-like numbers
    all_321_like_numbers = generate_321_like_numbers()
    
    # Output the K-th smallest 321-like number
    print(all_321_like_numbers[K-1])

if __name__ == "__main__":
    main()