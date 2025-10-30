def max_head_height(N, giants):
    # Sort giants based on the difference between head height and shoulder height (B_i - A_i)
    giants.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    # Initialize the total height from the ground
    total_height = 0
    
    # Place each giant in the sorted order
    for A_i, B_i in giants:
        total_height += A_i  # Add the shoulder height of the current giant
        # The head height of the current giant is not needed for the next iteration
    
    # Add the head height of the topmost giant
    total_height += giants[-1][1]  # The head height of the last giant in the sorted order
    
    return total_height

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    giants = [tuple(map(int, line.split())) for line in data[1:N+1]]
    
    result = max_head_height(N, giants)
    print(result)

if __name__ == "__main__":
    main()