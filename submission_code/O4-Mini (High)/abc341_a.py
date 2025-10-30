def main():
    import sys
    
    # Read input
    data = sys.stdin.readline().strip()
    N = int(data)
    
    # We need N zeros and N+1 ones alternating starting with '1'
    # That string is simply "10" repeated N times, plus a final '1'
    result = "10" * N + "1"
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()