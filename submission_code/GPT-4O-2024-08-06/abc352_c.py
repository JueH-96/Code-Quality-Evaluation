# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    giants = []
    
    index = 1
    for i in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        giants.append((A, B))
        index += 2
    
    # Sort giants by the difference B_i - A_i in descending order
    giants.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    
    # Calculate the maximum possible height
    current_height = 0
    for i in range(N):
        A, B = giants[i]
        current_height += A
    
    # Add the head height of the last giant in the sorted order
    max_height = current_height - giants[-1][0] + giants[-1][1]
    
    print(max_height)

if __name__ == "__main__":
    main()