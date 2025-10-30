import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        
        A = data[index:index + N]
        index += N
        B = data[index:index + N]
        index += N
        
        # We can use a dictionary to map each value in B to its indices
        target_positions = {}
        for i, value in enumerate(B):
            if value in target_positions:
                target_positions[value].append(i)
            else:
                target_positions[value] = [i]
        
        # Now we need to check if we can transform A into B
        possible = True
        source_positions = {}
        
        for i, value in enumerate(A):
            if value in source_positions:
                source_positions[value].append(i)
            else:
                source_positions[value] = [i]
        
        # Check for each value in B if it can be matched
        for value, positions in target_positions.items():
            if value not in source_positions:
                possible = False
                break
            
            source_pos = source_positions[value]
            source_idx = 0
            target_idx = 0
            
            # We need to match each position in positions with some position in source_pos
            while target_idx < len(positions):
                while (source_idx < len(source_pos) and 
                       (positions[target_idx] < source_pos[source_idx] - K or 
                        positions[target_idx] > source_pos[source_idx] + K)):
                    source_idx += 1
                
                if source_idx == len(source_pos):
                    possible = False
                    break
                
                target_idx += 1
        
        if possible:
            results.append("Yes")
        else:
            results.append("No")
    
    sys.stdout.write("
".join(results) + "
")