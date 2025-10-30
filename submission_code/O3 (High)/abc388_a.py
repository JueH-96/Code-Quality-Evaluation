import sys

def main():
    S = sys.stdin.readline().strip()   # Read the input string
    # Output the first character followed by "UPC"
    print(S[0] + "UPC")

if __name__ == "__main__":
    main()