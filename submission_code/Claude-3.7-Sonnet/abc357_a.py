def solve():
    # Read input
    N, M = map(int, input().split())
    H = list(map(int, input().split()))
    
    available_disinfectant = M
    aliens_count = 0
    
    for i in range(N):
        if available_disinfectant >= H[i]:
            # Alien can disinfect all hands
            available_disinfectant -= H[i]
            aliens_count += 1
        else:
            # Alien uses up remaining disinfectant but cannot disinfect all hands
            break  # No need to process further
    
    return aliens_count

print(solve())