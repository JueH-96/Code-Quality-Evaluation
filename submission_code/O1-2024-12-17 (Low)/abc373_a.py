def main():
    # Read 12 strings from standard input
    strings = [input().strip() for _ in range(12)]
    
    # Count how many strings S_i have length i
    count = sum(1 for i, s in enumerate(strings, start=1) if len(s) == i)
    
    # Output the result
    print(count)

# Call the main function
if __name__ == "__main__":
    main()