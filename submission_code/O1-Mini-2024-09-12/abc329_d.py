import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    votes = list(map(int, data[2:2+M]))
    counts = [0] * (N + 1)
    current_max = 0
    winner = 1  # Initialize with the smallest candidate number
    outputs = []
    for vote in votes:
        counts[vote] += 1
        if counts[vote] > current_max:
            current_max = counts[vote]
            winner = vote
        elif counts[vote] == current_max and vote < winner:
            winner = vote
        outputs.append(str(winner))
    print('
'.join(outputs))

if __name__ == "__main__":
    main()