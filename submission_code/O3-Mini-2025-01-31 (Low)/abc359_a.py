def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # The first value is N
    n = int(input_data[0])
    # The remaining values are the strings S_i
    strings = input_data[1:]
    # Count the number of occurrences of "Takahashi"
    count = strings.count("Takahashi")
    print(count)
    
if __name__ == '__main__':
    main()