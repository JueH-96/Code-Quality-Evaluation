# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Count the number of '@' in the string S
    cookie_count = S.count('@')
    
    # After D days, D cookies will be eaten
    remaining_cookies = cookie_count - D
    
    # The number of empty boxes will be the total number of boxes minus the remaining cookies
    empty_boxes = N - remaining_cookies
    
    print(empty_boxes)

if __name__ == "__main__":
    main()