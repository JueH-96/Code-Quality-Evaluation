# YOUR CODE HERE
def main():
    import sys
    input_data = sys.stdin.read().strip()
    surname, first_name = input_data.split()
    print(f"{surname} san")

if __name__ == "__main__":
    main()