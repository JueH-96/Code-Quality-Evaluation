def main():
    import sys
    input_line = sys.stdin.readline
    x = int(input_line().strip())
    
    # Using the identity: ceil(x/10) = -(-x // 10)
    result = -(-x // 10)
    print(result)

# Call main to execute the solution
if __name__ == "__main__":
    main()