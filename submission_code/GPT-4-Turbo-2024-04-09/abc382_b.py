# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Convert the string S into a list of characters for easier manipulation
    boxes = list(S)
    
    # We need to eat D cookies from the rightmost positions
    cookies_eaten = 0
    
    # Start from the rightmost box and eat cookies
    for i in range(N - 1, -1, -1):
        if boxes[i] == '@':
            boxes[i] = '.'
            cookies_eaten += 1
            if cookies_eaten == D:
                break
    
    # Join the list back into a string
    result = ''.join(boxes)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()