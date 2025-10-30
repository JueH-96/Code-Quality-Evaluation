def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    votes = list(map(int, input_data[2:]))

    counts = [0] * N  # counts[i] will store number of votes for candidate i+1
    current_winner = None
    max_votes = 0

    for i in range(M):
        candidate = votes[i] - 1
        counts[candidate] += 1

        if counts[candidate] > max_votes:
            max_votes = counts[candidate]
            current_winner = candidate
        elif counts[candidate] == max_votes:
            if candidate < current_winner:
                current_winner = candidate

        print(current_winner + 1)

def main():
    solve()

if __name__ == "__main__":
    main()