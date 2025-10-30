def solve():
    import sys
    S = sys.stdin.readline().strip()
    
    # Create a dictionary that maps each character to its coordinate (1-based index in S)
    position = {}
    for i, ch in enumerate(S, start=1):
        position[ch] = i
    
    order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Start from 'A' and sum the distances between consecutive characters
    total_distance = 0
    current_coordinate = position['A']
    
    for char in order[1:]:  # from 'B' to 'Z'
        next_coordinate = position[char]
        total_distance += abs(next_coordinate - current_coordinate)
        current_coordinate = next_coordinate
    
    print(total_distance)

def main():
    solve()
    
if __name__ == "__main__":
    main()