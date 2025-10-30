def solve():
    N, M = map(int, input().split())
    testimonies = []
    for _ in range(M):
        a, b, c = map(int, input().split())
        testimonies.append((a-1, b-1, c))
    
    # Try all possible combinations of confused villagers
    for mask in range(1 << N):
        confused = [bool(mask & (1 << i)) for i in range(N)]
        
        if check_consistency(N, testimonies, confused):
            result = ''.join('1' if confused[i] else '0' for i in range(N))
            return result
    
    return "-1"

def check_consistency(N, testimonies, confused):
    # Try all possible honest/liar assignments
    for honest_mask in range(1 << N):
        honest = [bool(honest_mask & (1 << i)) for i in range(N)]
        
        # Check if this assignment satisfies all testimonies
        valid = True
        for a, b, c in testimonies:
            # Does villager a tell the truth?
            tells_truth = (honest[a] and not confused[a]) or (not honest[a] and confused[a])
            
            # What does the testimony claim?
            claims_b_honest = (c == 0)
            
            # Check consistency
            if tells_truth:
                # Truth teller: testimony should match reality
                if claims_b_honest != honest[b]:
                    valid = False
                    break
            else:
                # Liar: testimony should be opposite of reality
                if claims_b_honest == honest[b]:
                    valid = False
                    break
        
        if valid:
            return True
    
    return False

print(solve())