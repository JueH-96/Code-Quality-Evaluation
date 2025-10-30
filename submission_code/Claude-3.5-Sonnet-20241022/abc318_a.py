N, M, P = map(int, input().split())

if M > N:
    print(0)
else:
    # Number of full moons = 1 + number of complete P-day cycles that fit before N
    num_cycles = (N - M) // P
    total_days = num_cycles + 1
    print(total_days)