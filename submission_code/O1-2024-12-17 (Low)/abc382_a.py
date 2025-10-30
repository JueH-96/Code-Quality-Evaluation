def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, D = map(int, data[:2])
    S = data[2]
    
    # Count how many boxes are initially empty ('.')
    empty_count = S.count('.')
    # The answer is the sum of the initially empty boxes and the D eaten cookies
    print(empty_count + D)

# Do not forget to call the main function
if __name__ == "__main__":
    main()