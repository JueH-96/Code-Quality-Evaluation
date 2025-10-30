n, m = map(int, input().split())
h = list(map(int, input().split()))

remaining_disinfectant = m
count_disinfected_aliens = 0

for num_hands in h:
    if remaining_disinfectant >= num_hands:
        remaining_disinfectant -= num_hands
        count_disinfected_aliens += 1
    else:
        remaining_disinfectant -= remaining_disinfectant # Use up the remaining disinfectant
        break # No more aliens can disinfect fully

print(count_disinfected_aliens)