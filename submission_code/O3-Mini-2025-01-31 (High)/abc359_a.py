def main():
    import sys
    input_data = sys.stdin.read().split()
    # The first token is the number of strings, N.
    n = int(input_data[0])
    # The remaining tokens are the strings S_1, S_2, ..., S_N.
    strings = input_data[1:]
    
    # Count the number of times "Takahashi" appears.
    count_takahashi = sum(1 for s in strings if s == "Takahashi")
    
    # Print the result.
    sys.stdout.write(str(count_takahashi))

if __name__ == "__main__":
    main()