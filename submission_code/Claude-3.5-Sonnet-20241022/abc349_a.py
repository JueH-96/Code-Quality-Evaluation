N = int(input())
A = list(map(int, input().split()))

# The sum of all scores must be 0 since each game adds 1 to one player and subtracts 1 from another
# So if we know N-1 scores, we can find the Nth score
total_known_scores = sum(A)
person_N_score = -total_known_scores

print(person_N_score)