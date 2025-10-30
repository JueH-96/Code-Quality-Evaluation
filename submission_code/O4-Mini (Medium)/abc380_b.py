def main():
    import sys
    s = sys.stdin.readline().strip()
    # Remove the first and last '|' and split by '|' to get groups of dashes
    parts = s[1:-1].split('|')
    # Convert each group to its length (number of '-')
    result = [str(len(p)) for p in parts]
    # Print the sequence
    print(" ".join(result))

# Call main to execute
main()