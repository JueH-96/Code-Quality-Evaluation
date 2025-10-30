def solve():
    n = int(input())
    stages = []
    for _ in range(n - 1):
        stages.append(list(map(int, input().split())))

    min_times = {i: float('inf') for i in range(1, n + 1)}
    min_times[1] = 0
    
    playable_stages = [1]

    while playable_stages:
        current_stage = playable_stages.pop(0)
        
        if current_stage == n:
            continue

        a_cost, b_cost, x_stage = stages[current_stage - 1]
        
        next_stage_a = current_stage + 1
        if next_stage_a <= n:
            if min_times[next_stage_a] > min_times[current_stage] + a_cost:
                min_times[next_stage_a] = min_times[current_stage] + a_cost
                if next_stage_a not in playable_stages:
                    playable_stages.append(next_stage_a)
        
        if min_times[x_stage] > min_times[current_stage] + b_cost:
            min_times[x_stage] = min_times[current_stage] + b_cost
            if x_stage not in playable_stages:
                playable_stages.append(x_stage)
        
        playable_stages.sort()

    print(min_times[n])

solve()