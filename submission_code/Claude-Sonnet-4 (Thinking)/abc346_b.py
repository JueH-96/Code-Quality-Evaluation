pattern = "wbwbwwbwbwbw"
cycle_w = 7
cycle_b = 5

def solve(W, B):
    for start in range(12):
        w_count = 0
        b_count = 0
        for length in range(1, 13):
            if pattern[(start + length - 1) % 12] == 'w':
                w_count += 1
            else:
                b_count += 1
            
            if W >= w_count and B >= b_count:
                remaining_w = W - w_count
                remaining_b = B - b_count
                
                if remaining_w % cycle_w == 0 and remaining_b % cycle_b == 0:
                    cycles_w = remaining_w // cycle_w
                    cycles_b = remaining_b // cycle_b
                    if cycles_w == cycles_b:
                        return "Yes"
    
    return "No"

W, B = map(int, input().split())
print(solve(W, B))