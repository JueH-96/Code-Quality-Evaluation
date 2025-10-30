# Read input
N_T_P = input().split()
N = int(N_T_P[0])
T = int(N_T_P[1])
P = int(N_T_P[2])
L = list(map(int, input().split()))

# Initial count of people with hair length >= T
initial_count = sum(1 for l in L if l >= T)

# If already enough people have hair length >= T
if initial_count >= P:
    print(0)
else:
    # Calculate required days for those with L_i < T
    required_days = [T - l for l in L if l < T]
    # Sort the required days
    required_days.sort()
    # Find the (P - initial_count - 1)-th day
    need = P - initial_count
    D = required_days[need - 1]
    print(D)