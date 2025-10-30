import sys

def main() -> None:
    # Read the 6-digit number as a string
    s = sys.stdin.readline().strip()
    
    # Count occurrences of '1', '2', and '3'
    c1 = s.count('1')
    c2 = s.count('2')
    c3 = s.count('3')
    
    # Check the required counts
    if c1 == 1 and c2 == 2 and c3 == 3:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()