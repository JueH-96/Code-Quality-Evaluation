def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    votes = list(map(int, input_data[2:]))

    counts = [0] * (N + 1)  # Index 0 unused
    current_winner = 0
    max_votes = 0

    for i in range(M):
        candidate = votes[i]
        counts[candidate] += 1

        if counts[candidate] > max_votes:
            max_votes = counts[candidate]
            current_winner = candidate
        elif counts[candidate] == max_votes and candidate < current_winner:
            current_winner = candidate

        print(current_winner)

# Do not forget to call the main function
if __name__ == "__main__":
    main()