def main():
    import sys
    # Read all input numbers as strings and split them
    data = sys.stdin.read().strip().split()
    # Convert the input strings to integers
    numbers = list(map(int, data))
    # Reverse the list of numbers
    numbers.reverse()
    # Print each number in the reversed list on a separate line
    for num in numbers:
        print(num)
        
main()