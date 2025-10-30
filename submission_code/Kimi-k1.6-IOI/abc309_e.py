import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    parent = [0] * (N + 1)  # 1-based
    for i in range(2, N+1):
        parent[i] = int(input[ptr])
        ptr +=1
    
    # Compute depth
    depth = [0] * (N +1)
    depth[1] = 0
    for i in range(2, N+1):
        depth[i] = depth[parent[i]] +1
    
    # Process insurances
    max_y = [-float('inf')] * (N +1)
    for _ in range(M):
        x = int(input[ptr])
        ptr +=1
        y = int(input[ptr])
        ptr +=1
        if y > max_y[x]:
            max_y[x] = y
    
    # Compute max_along_path
    max_along_path = [-float('inf')] * (N +1)
    count =0
    # Root node
    max_along_path[1] = depth[1] + max_y[1]
    if max_along_path[1] >= depth[1]:
        count +=1
    
    for i in range(2, N+1):
        current_value = depth[i] + max_y[i]
        max_along_path[i] = max(max_along_path[parent[i]], current_value)
        if max_along_path[i] >= depth[i]:
            count +=1
    
    print(count)

if __name__ == "__main__":
    main()