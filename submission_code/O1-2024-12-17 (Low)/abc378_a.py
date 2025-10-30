def main():
    import sys
    # Read the four ball colors
    A = list(map(int, sys.stdin.readline().split()))
    
    # Count the occurrences of each color
    from collections import Counter
    color_count = Counter(A)
    
    # We can form a pair for each two occurrences of the same color
    operations = sum(count // 2 for count in color_count.values())
    
    # Since we only have 4 balls total, the maximum number of pairs (operations) is 2
    operations = min(operations, 2)
    
    # Output the result
    print(operations)

# Do not forget to call main()
if __name__ == "__main__":
    main()