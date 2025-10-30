# YOUR CODE HERE
def solve_sushi_problem():
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Initialize an array to keep track of the next person to check for each sushi
    next_person = [0] * M

    # Process each sushi
    for j in range(M):
        sushi_eaten = False
        for i in range(next_person[j], N):
            if B[j] >= A[i]:
                print(i + 1)  # Person i+1 eats the sushi
                sushi_eaten = True
                # Update next_person for all subsequent sushi
                for k in range(j + 1, M):
                    next_person[k] = max(next_person[k], i + 1)
                break
        if not sushi_eaten:
            print(-1)  # Nobody eats the sushi

# Run the solver
solve_sushi_problem()