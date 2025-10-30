# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Convert the string S into a list of characters for easy manipulation
    boxes = list(S)
    
    # We need to remove D cookies from the rightmost side
    cookies_to_remove = D
    
    # Traverse from the rightmost side and remove cookies
    for i in range(N-1, -1, -1):
        if cookies_to_remove == 0:
            break
        if boxes[i] == '@':
            boxes[i] = '.'
            cookies_to_remove -= 1
    
    # Convert the list back to a string and print the result
    result = ''.join(boxes)
    print(result)

if __name__ == "__main__":
    main()