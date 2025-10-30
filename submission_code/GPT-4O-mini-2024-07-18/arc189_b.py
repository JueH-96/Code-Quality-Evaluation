def minimize_sum_of_coordinates(N, coordinates):
    # The minimum sum can be achieved by pairing the coordinates optimally.
    # We will use the first and last coordinates to determine the midpoints.
    
    # The optimal positions after operations will be:
    # - The first piece remains at its position.
    # - The last piece remains at its position.
    # - The second and second last pieces will be adjusted to be symmetric around the midpoint
    #   of the first and last pieces.
    
    # Calculate the midpoint of the first and last pieces
    first = coordinates[0]
    last = coordinates[-1]
    midpoint = (first + last) / 2
    
    # Calculate the new positions for the second and second last pieces
    second = coordinates[1]
    second_last = coordinates[-2]
    
    # New positions for the second and second last pieces
    new_second = midpoint + (midpoint - second)
    new_second_last = midpoint - (second_last - midpoint)
    
    # Calculate the new sum
    new_sum = first + new_second + new_second_last + last
    
    # The remaining pieces (if any) will remain at their original positions
    for i in range(2, N-2):
        new_sum += coordinates[i]
    
    return int(new_sum)

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    coordinates = list(map(int, data[1:]))
    
    result = minimize_sum_of_coordinates(N, coordinates)
    print(result)

if __name__ == "__main__":
    main()