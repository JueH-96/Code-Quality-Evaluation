def max_height():
    n = int(input())
    giants = []
    
    for _ in range(n):
        a, b = map(int, input().split())
        giants.append((a, b))
    
    # Calculate the total of all shoulder heights
    total_shoulder_height = sum(g[0] for g in giants)
    
    # Calculate the maximum possible head height
    max_head_height = max(total_shoulder_height - giants[i][0] + giants[i][1] for i in range(n))
    
    return max_head_height

print(max_height())