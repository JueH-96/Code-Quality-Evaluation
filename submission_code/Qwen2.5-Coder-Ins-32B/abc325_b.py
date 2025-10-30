# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    data = []
    index = 1
    for _ in range(N):
        W = int(input[index])
        X = int(input[index + 1])
        data.append((W, X))
        index += 2

    max_participants = 0
    for utc_start in range(24):
        participants = 0
        for W, X in data:
            local_start = (utc_start - X) % 24
            if 9 <= local_start <= 17:
                participants += W
        max_participants = max(max_participants, participants)

    print(max_participants)

if __name__ == "__main__":
    main()