import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    left_keys = []
    right_keys = []
    idx = 1
    for _ in range(N):
        key = int(data[idx])
        hand = data[idx + 1]
        idx += 2
        if hand == 'L':
            left_keys.append(key)
        else:            # hand == 'R'
            right_keys.append(key)

    def path_cost(seq):
        """sum of absolute differences between consecutive keys"""
        return sum(abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1))

    fatigue = path_cost(left_keys) + path_cost(right_keys)
    print(fatigue)

if __name__ == "__main__":
    main()