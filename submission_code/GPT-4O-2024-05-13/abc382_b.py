# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Convert the string S to a list for easier manipulation
    boxes = list(S)
    
    # Takahashi eats one cookie per day for D days
    for _ in range(D):
        # Find the rightmost box with a cookie
        for i in range(N-1, -1, -1):
            if boxes[i] == '@':
                # Eat the cookie
                boxes[i] = '.'
                break
    
    # Convert the list back to a string
    result = ''.join(boxes)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()