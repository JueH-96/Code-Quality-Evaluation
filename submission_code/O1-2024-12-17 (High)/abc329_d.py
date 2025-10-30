# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    votes = list(map(int, data[2:]))

    freq = [0] * N
    winner = 1
    maxfreq = 0
    results = []

    for v in votes:
        freq[v - 1] += 1
        if freq[v - 1] > maxfreq:
            maxfreq = freq[v - 1]
            winner = v
        elif freq[v - 1] == maxfreq and v < winner:
            winner = v
        results.append(str(winner))

    print("
".join(results))

# Do not forget to call main()
main()