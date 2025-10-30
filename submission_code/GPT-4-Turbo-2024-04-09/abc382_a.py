# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Count the number of cookies initially
    initial_cookies = S.count('@')
    
    # Calculate the number of empty boxes after D days
    # Since D cookies will be eaten, the number of remaining cookies is initial_cookies - D
    remaining_cookies = initial_cookies - D
    
    # The number of empty boxes is total boxes minus the remaining cookies
    empty_boxes = N - remaining_cookies
    
    # Output the result
    print(empty_boxes)

if __name__ == "__main__":
    main()