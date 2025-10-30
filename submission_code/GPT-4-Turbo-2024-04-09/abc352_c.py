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
    
    # Sort the giants by a strategy that maximizes the height of the topmost giant's head
    # We sort by the difference B - A in descending order because we want the giant which
    # adds the least to the height due to its own head to be at the bottom.
    giants.sort(key=lambda x: (x[0] - x[1], -x[0]))
    
    # Calculate the maximum height of the topmost giant's head
    max_height = 0
    current_shoulder_height = 0
    
    for A, B in giants:
        current_head_height = current_shoulder_height + B
        max_height = max(max_height, current_head_height)
        current_shoulder_height += A
    
    print(max_height)

if __name__ == "__main__":
    main()