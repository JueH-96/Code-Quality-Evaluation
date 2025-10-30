def main():
    import sys
    from collections import Counter
    
    # Read inputs
    A_1, A_2, A_3, A_4 = map(int, sys.stdin.read().split())
    
    # Count occurrences of each color
    counter = Counter([A_1, A_2, A_3, A_4])
    
    # Calculate total pairs
    pairs = sum(value // 2 for value in counter.values())
    
    # Print result
    print(pairs)

# Do not forget to call main function
main()