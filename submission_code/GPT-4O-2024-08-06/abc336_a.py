# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Construct the Dragon String of level N
    dragon_string = "L" + "o" * N + "ng"
    
    # Print the result
    print(dragon_string)

if __name__ == "__main__":
    main()