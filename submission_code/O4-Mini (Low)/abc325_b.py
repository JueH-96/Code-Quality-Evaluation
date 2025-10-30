import sys

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # Next 2*N values are W_i, X_i pairs
    WX = list(map(int, data[1:]))

    # Prepare lists of W and X
    W = WX[0::2]
    X = WX[1::2]

    # For each possible UTC meeting start time t in [0,23],
    # compute how many employees can attend
    max_attendees = 0
    for t in range(24):
        current = 0
        for wi, xi in zip(W, X):
            local_start = (t + xi) % 24
            # Meeting is from local_start to local_start+1
            # Must lie entirely in [9,18], so local_start in [9,17]
            if 9 <= local_start <= 17:
                current += wi
        if current > max_attendees:
            max_attendees = current

    print(max_attendees)

if __name__ == "__main__":
    main()