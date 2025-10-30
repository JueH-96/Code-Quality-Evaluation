N = int(input())
H = list(map(int, input().split()))

# Simulate the process and track when each position becomes positive
result = [0] * N
seen = set()

def get_state(A):
    return tuple(A)

def simulate():
    A = [0] * (N + 1)
    ops = 0
    
    while True:
        # All positions filled
        if all(result[i] != 0 for i in range(N)):
            break
            
        ops += 1
        # Step 1: Increase A[0]
        A[0] += 1
        
        # Step 2: Process each position
        for i in range(1, N + 1):
            if A[i-1] > A[i] and A[i-1] > H[i-1]:
                A[i-1] -= 1
                A[i] += 1
                
        # Record first time each position becomes positive
        for i in range(N):
            if result[i] == 0 and A[i+1] > 0:
                result[i] = ops
        
        # Check for cycle
        state = get_state(A)
        if state in seen:
            break
        seen.add(state)

simulate()
print(*result)