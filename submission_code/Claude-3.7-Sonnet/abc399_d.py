def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        arr = list(map(int, input().split()))
        
        # Find positions of each number
        positions = {}
        for i, num in enumerate(arr):
            if num not in positions:
                positions[num] = []
            positions[num].append(i)
        
        count = 0
        # Check each pair (a,b) where a < b
        for a in range(1, n+1):
            for b in range(a+1, n+1):
                if is_valid_pair(positions[a], positions[b]):
                    count += 1
        
        print(count)

def is_valid_pair(a_positions, b_positions):
    # Check if occurrences of a are not adjacent
    if abs(a_positions[0] - a_positions[1]) == 1:
        return False
    
    # Check if occurrences of b are not adjacent
    if abs(b_positions[0] - b_positions[1]) == 1:
        return False
    
    # Sort all four positions
    all_positions = sorted(a_positions + b_positions)
    
    # Check if they can be partitioned into two pairs of consecutive positions
    # This happens when position[1]-position[0]=1 and position[3]-position[2]=1
    return all_positions[1] - all_positions[0] == 1 and all_positions[3] - all_positions[2] == 1

if __name__ == "__main__":
    solve()