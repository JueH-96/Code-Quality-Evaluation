import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, T = map(int, input().split())
    # Scores of players, initialized to 0
    scores = [0] * (N + 1)
    # Frequency map: score_value -> count of players with that score
    freq = {0: N}
    distinct = 1  # Initially only score 0

    out = []
    for _ in range(T):
        a, b = map(int, input().split())
        old = scores[a]
        new = old + b

        # Decrease count of the old score
        cnt_old = freq[old]
        if cnt_old == 1:
            # remove old score entirely
            del freq[old]
            distinct -= 1
        else:
            freq[old] = cnt_old - 1

        # Increase count of the new score
        cnt_new = freq.get(new, 0)
        if cnt_new == 0:
            distinct += 1
        freq[new] = cnt_new + 1

        # Update the player's score
        scores[a] = new

        # Record the number of distinct scores
        out.append(str(distinct))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()