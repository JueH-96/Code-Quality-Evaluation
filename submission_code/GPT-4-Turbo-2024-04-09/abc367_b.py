import sys
input = sys.stdin.read

def main():
    x = input().strip()
    # Convert the string input to a float and then back to a string
    # This will automatically remove unnecessary trailing zeros and decimal points
    formatted_x = str(float(x))
    print(formatted_x)

if __name__ == "__main__":
    main()