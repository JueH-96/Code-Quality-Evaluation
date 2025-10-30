def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Check if "ab" or "ba" occurs in S
    if "ab" in S or "ba" in S:
        print("Yes")
    else:
        print("No")

# Call main() to run the solution
main()